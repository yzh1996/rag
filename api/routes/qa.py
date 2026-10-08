"""
问答API
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from api.dependencies import get_rag_chain_dep
from core.rag_chain import RAGChain

router = APIRouter(prefix="/qa", tags=["智能问答"])


class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"


class SourceInfo(BaseModel):
    source: str
    page: int = None
    content: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list
    intent: str


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(
    request: QuestionRequest,
    rag_chain: RAGChain = Depends(get_rag_chain_dep)
):
    """
    提交问题并获取答案
    """
    if not request.question.strip():
        raise HTTPException(400, "问题不能为空")

    result = rag_chain.ask(request.question, request.session_id)
    return AnswerResponse(
        answer=result["answer"],
        sources=[SourceInfo(**s) for s in result["sources"]],
        intent=result["intent"]
    )



@router.post("/clear_memory")
async def clear_conversation_memory(
    session_id: str = Query("default", description="会话ID"),
    rag_chain: RAGChain = Depends(get_rag_chain_dep)
):
    """
    清除对话记忆
    """
    rag_chain.clear_session(session_id)
    return {"status": "success", "message": "对话记忆已清除"}


@router.get("/history")
async def get_chat_history(
    session_id: str = Query("default", description="会话ID"),
    rag_chain: RAGChain = Depends(get_rag_chain_dep)
):
    """
    获取当前聊天历史
    """
    history = rag_chain.get_chat_history(session_id)
    return {"history": history}