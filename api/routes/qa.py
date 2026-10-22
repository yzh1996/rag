"""问答API。"""
from typing import Literal
import asyncio
import json
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeout

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.database import get_db
from core.langfuse_tracker import LangfuseTracker
from core.phase1_services import get_interaction, save_feedback
from api.dependencies import get_rag_chain_dep
from core.rag_chain import RAGChain
from core.auth import AuthUser, get_current_user
from core.feedback_service import assert_interaction_access
from core.metrics import increment
from sqlalchemy import func, select
from core.db_models import QAInteraction
from core.telemetry import span

router = APIRouter(prefix="/qa", tags=["智能问答"])


class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"


class SourceInfo(BaseModel):
    source: str
    page: int = None
    content: str
    document_id: str | None = None
    citation_id: int | None = None


class AnswerResponse(BaseModel):
    answer: str
    sources: list
    intent: str
    interaction_id: str | None = None
    trace_id: str | None = None
    evidence_sufficient: bool | None = None
    citation_status: str | None = None
    response_type: str | None = None
    query_details: dict | None = None


class FeedbackRequest(BaseModel):
    interaction_id: str
    rating: Literal["positive", "negative"]
    category: str | None = None
    comment: str | None = None
    corrected_answer: str | None = Field(default=None, max_length=20000)
    is_resolved: bool | None = None
    handoff_requested: bool = False


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(
    request: QuestionRequest,
    rag_chain: RAGChain = Depends(get_rag_chain_dep),
    user: AuthUser = Depends(get_current_user),
):
    """
    提交问题并获取答案。

    这一层只负责 HTTP 相关工作：检查请求、取得当前用户、调用 RAGChain，
    不在接口里实现检索和大模型生成，具体业务交给 core/rag_chain.py。
    """
    if not request.question.strip():
        raise HTTPException(400, "问题不能为空")

    # ask() 会完成一次完整问答，并返回答案、来源、意图等结构化结果。
    result = rag_chain.ask(request.question, request.session_id, user=user)
    return AnswerResponse(
        answer=result["answer"],
        sources=[SourceInfo(**s) for s in result["sources"]],
        intent=result["intent"],
        interaction_id=result.get("interaction_id"),
        trace_id=result.get("trace_id"),
        evidence_sufficient=result.get("evidence_sufficient"),
        citation_status=result.get("citation_status"),
        response_type=result.get("response_type"),
        query_details=result.get("query_details"),
    )


@router.post("/ask/stream")
async def stream_question(
    request: QuestionRequest,
    http_request: Request,
    rag_chain: RAGChain = Depends(get_rag_chain_dep),
    user: AuthUser = Depends(get_current_user),
):
    """通过 SSE 发送进度、正文片段及最终结果，沿用现有身份与知识权限。

    SSE 是一种服务器主动推送文字的方式，适合把模型生成中的每一小段答案
    及时发给浏览器，而不用等整段答案生成完毕。
    """
    if not request.question.strip():
        raise HTTPException(400, "问题不能为空")
    started_at = getattr(http_request.state, "started_at", time.perf_counter())

    async def events():
        # 生成模型通常是同步迭代器，因此放到工作线程，避免阻塞 FastAPI 的异步事件循环。
        loop = asyncio.get_running_loop()
        outbox = asyncio.Queue(maxsize=8)
        cancelled = threading.Event()

        def send(event):
            # 有界队列限制慢客户端占用的内存，断线时解除工作线程的发送等待。
            future = asyncio.run_coroutine_threadsafe(outbox.put(event), loop)
            try:
                while not cancelled.is_set():
                    try:
                        future.result(timeout=0.1)
                        return True
                    except FutureTimeout:
                        continue
                return False
            finally:
                if not future.done():
                    future.cancel()

        def produce():
            try:
                # 整个模型迭代在同一工作线程中完成，保留追踪上下文且不阻塞异步接口。
                with span("rag.ask.stream"):
                    iterator = rag_chain.stream_answer(request.question, request.session_id, user=user,
                        started_at=started_at, cancelled=cancelled)
                    try:
                        for event in iterator:
                            if cancelled.is_set() or not send(event):
                                break
                    finally:
                        iterator.close()
            except Exception:
                if not cancelled.is_set():
                    send({"event": "error", "data": {"answer": "回答生成失败，请稍后重试"}})
            finally:
                if not cancelled.is_set():
                    send(None)

        worker = asyncio.create_task(asyncio.to_thread(produce))
        try:
            while True:
                try:
                    event = await asyncio.wait_for(outbox.get(), timeout=15)
                except asyncio.TimeoutError:
                    # 心跳只用于保持连接，不是答案文字，不计入首字耗时。
                    yield ": ping\n\n"
                    continue
                if event is None:
                    break
                payload = json.dumps(event["data"], ensure_ascii=False)
                yield f"event: {event['event']}\ndata: {payload}\n\n"
        finally:
            cancelled.set()
            # 上游生成可能正在等网络；让它返回后自行关闭，避免断线阻塞请求退出。
            worker.add_done_callback(lambda task: None if task.cancelled() else task.exception())

    return StreamingResponse(events(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest, db: Session = Depends(get_db), user: AuthUser = Depends(get_current_user)):
    """保存用户对单次问答的反馈，并同步到Langfuse。"""
    interaction = get_interaction(db, request.interaction_id)
    if not interaction:
        raise HTTPException(404, "问答记录不存在或已过期")
    assert_interaction_access(interaction, user)
    feedback = save_feedback(
        db,
        interaction_id=request.interaction_id,
        rating=request.rating,
        category=request.category,
        comment=request.comment,
        corrected_answer=request.corrected_answer,
        is_resolved=request.is_resolved,
        handoff_requested=request.handoff_requested,
    )
    increment("feedback_total")
    increment("feedback_negative_total" if request.rating == "negative" else "feedback_positive_total")
    if request.handoff_requested:
        increment("handoff_requests_total")
    if interaction.trace_id:
        LangfuseTracker(
            interaction.session_id,
            trace_id=interaction.trace_id,
        ).score_feedback(request.rating, request.comment)
    return {"status": "success", "feedback_id": feedback.id}



@router.post("/clear_memory")
async def clear_conversation_memory(
    session_id: str = Query("default", description="会话ID"),
    rag_chain: RAGChain = Depends(get_rag_chain_dep),
    user: AuthUser = Depends(get_current_user),
):
    """
    清除对话记忆
    """
    rag_chain.clear_session(session_id, user=user)
    return {"status": "success", "message": "对话记忆已清除"}


@router.get("/history")
async def get_chat_history(
    session_id: str = Query("default", description="会话ID"),
    rag_chain: RAGChain = Depends(get_rag_chain_dep),
    user: AuthUser = Depends(get_current_user),
):
    """
    获取当前聊天历史
    """
    history = rag_chain.get_chat_history(session_id, user=user)
    return {"history": history}


@router.get("/sessions")
async def get_saved_sessions(db: Session = Depends(get_db), user: AuthUser = Depends(get_current_user)):
    rows = db.execute(select(QAInteraction.session_id, func.max(QAInteraction.created_at).label("updated_at"))
        .where(QAInteraction.username == user.username, QAInteraction.tenant_id == user.tenant_id,
            QAInteraction.status == "completed").group_by(QAInteraction.session_id)
        .order_by(func.max(QAInteraction.created_at).desc()).limit(50)).all()
    items = []
    for session_id, updated_at in rows:
        title = db.scalar(select(QAInteraction.question).where(QAInteraction.session_id == session_id,
            QAInteraction.username == user.username, QAInteraction.tenant_id == user.tenant_id)
            .order_by(QAInteraction.created_at.asc()).limit(1))
        items.append({"session_id": session_id, "name": (title or "历史会话")[:30], "updated_at": updated_at})
    return {"items": items}
