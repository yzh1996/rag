"""流式问答的隔离验证：无真实模型费用，不读写业务数据。"""
import json
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.runnables import RunnableLambda
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.dependencies import get_rag_chain_dep
from api.routes.qa import router
from config.settings import settings
from core import database, metrics
from core.auth import ANONYMOUS_USER, create_access_token
from core.db_models import QAInteraction
from core.intent_recognizer import IntentRecognizer
from core.intent_router import IntentPrediction, IntentRouter
from core.memory_manager import MemoryManager
from core.quality_guard import REFUSAL, Verdict
from core.rag_chain import RAGChain
from frontend.qa_stream import read_answer_events


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'stream.db'}", connect_args={"check_same_thread": False})
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "SessionLocal", sessionmaker(engine, expire_on_commit=False))
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", False)
    monkeypatch.setattr(settings, "INTENT_ROUTER_ENABLED", False)
    # Keep the legacy RAG/stream timing contract covered as an explicit rollback mode.
    monkeypatch.setattr(settings, "AGENT_ENABLED", False)
    monkeypatch.setattr(settings, "AUTH_REQUIRED", False)
    monkeypatch.setattr(settings, "QUALITY_GUARD_ENABLED", False)
    monkeypatch.setattr(settings, "REQUIRE_EVIDENCE", True)
    monkeypatch.setattr(settings, "ALLOW_GENERAL_ANSWERS", False)
    monkeypatch.setattr(settings, "EVIDENCE_MIN_SOURCES", 1)
    database.init_db()
    yield
    engine.dispose()


def make_chain(model=None, documents=None, retrieved=None):
    chain = object.__new__(RAGChain)
    chain.llm = model or FakeListChatModel(responses=["第一段。第二段。"])
    chain.intent_recognizer = IntentRecognizer()
    chain.memory_manager = MemoryManager()
    documents = [Document(page_content="报销需提交申请", metadata={"source": "policy.txt", "document_id": "doc"})] if documents is None else documents
    def search(query):
        if retrieved is not None:
            retrieved.append(query)
        return documents
    chain.retriever = SimpleNamespace(get_compresstion_retriever=lambda **kwargs: RunnableLambda(search))
    chain._rag_chain = chain._create_rag_chain()
    return chain


def make_app(chain):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_rag_chain_dep] = lambda: chain
    @app.middleware("http")
    async def timing(request, call_next):
        request.state.started_at = time.perf_counter()
        return await call_next(request)
    @app.get("/ping")
    def ping():
        return {"ok": True}
    return app


def first_count():
    return metrics.snapshot()["latency_ms"].get("qa_first_token", {}).get("count", 0)


def test_stream_is_incremental_and_saves_one_complete_answer(isolated):
    chain = make_chain()
    before = first_count()
    stream = chain.stream_answer("报销怎么申请", "session", user=ANONYMOUS_USER)
    first = next(event for event in stream if event["event"] == "delta")
    assert first["data"]["text"] == "第"
    assert first["data"]["first_token_ms"] >= 0
    with database.session_scope() as db:
        assert db.query(QAInteraction).count() == 0
    rest = list(stream)
    done = rest[-1]
    assert done["event"] == "done" and done["data"]["answer"] == "第一段。第二段。"
    assert done["data"]["first_token_ms"] == first["data"]["first_token_ms"]
    assert done["data"]["sources"][0]["document_id"] == "doc"
    assert first_count() == before + 1
    with database.session_scope() as db:
        row = db.query(QAInteraction).one()
        assert row.answer == done["data"]["answer"] and row.status == "completed"
        assert row.id == done["data"]["interaction_id"] and row.tenant_id == "default"
    assert chain.get_chat_history("session", user=ANONYMOUS_USER)[-1]["content"] == done["data"]["answer"]


def test_history_rewrite_is_used_for_search_but_never_sent_as_answer(isolated):
    retrieved = []
    chain = make_chain(FakeListChatModel(responses=["独立改写问题", "最终回答"]), retrieved=retrieved)
    with database.session_scope() as db:
        db.add(QAInteraction(session_id="session", username="anonymous", question="报销", answer="上一轮", status="completed"))
    events = list(chain.stream_answer("具体怎么申请", "session", user=ANONYMOUS_USER))
    assert retrieved == ["独立改写问题"]
    assert "".join(event["data"]["text"] for event in events if event["event"] == "delta") == "最终回答"


def test_first_token_ignores_empty_and_whitespace_and_includes_prior_wait(isolated, monkeypatch):
    import core.rag_chain as module
    chain = make_chain()
    def chunks(inputs, config):
        yield {"answer": ""}
        yield {"answer": " \n"}
        yield {"answer": "正文", "context": []}
        yield {"answer": "结尾"}
    monkeypatch.setattr(chain, "_create_rag_chain", lambda *args, **kwargs: RunnableLambda(chunks))
    monkeypatch.setattr(module, "time", SimpleNamespace(perf_counter=lambda: 3.0))
    before = first_count()
    events = list(chain.stream_answer("如何申请", started_at=1.5))
    delta = [event["data"] for event in events if event["event"] == "delta"]
    assert delta[0] == {"text": " \n正文", "first_token_ms": 1500.0}
    assert events[-1]["data"]["first_token_ms"] == 1500.0
    assert first_count() == before + 1
    assert "rag_qa_first_token_seconds_bucket" in metrics.prometheus_text()


def test_no_evidence_refuses_before_model_generation(isolated):
    class UnusedModel(FakeListChatModel):
        def _stream(self, *args, **kwargs):
            pytest.fail("证据不足时不应生成答案")
            yield
    events = list(make_chain(UnusedModel(responses=["unused"]), documents=[]).stream_answer("如何报销"))
    assert events[-1]["data"]["response_type"] == "refuse"
    assert events[-1]["data"]["answer"] == "根据现有知识库无法回答该问题"


@pytest.mark.parametrize("supported", [False, True])
def test_strict_review_finishes_before_any_answer_is_sent(isolated, monkeypatch, supported):
    monkeypatch.setattr(settings, "QUALITY_GUARD_ENABLED", True)
    reviewed = []
    class ReviewModel(FakeListChatModel):
        def with_structured_output(self, schema, **kwargs):
            def review(inputs):
                reviewed.append(True)
                return Verdict(safe=True, evidence_sufficient=True, all_claims_supported=supported, citations=[1])
            return RunnableLambda(review)
    chain = make_chain(ReviewModel(responses=["待审核答案 [1]"]))
    stream = chain.stream_answer("如何报销")
    first = next(event for event in stream if event["event"] == "delta")
    assert reviewed == [True]
    assert first["data"]["text"] == ("待审核答案 [1]" if supported else REFUSAL)
    done = list(stream)[-1]["data"]
    assert done["citation_status"] == ("verified" if supported else "rejected")


def test_clarification_is_timed_and_skips_model(isolated):
    chain = make_chain()
    chain._create_rag_chain = lambda *args, **kwargs: pytest.fail("澄清问题不应调用模型")
    before = first_count()
    events = list(chain.stream_answer("那个怎么弄？"))
    assert events[-1]["data"]["response_type"] == "clarify"
    assert first_count() == before + 1


@pytest.fixture
def model_router(isolated, monkeypatch):
    """只替换分类模型，仍通过真实路由器和问答编排验证处理结果。"""
    monkeypatch.setattr(settings, "INTENT_ROUTER_ENABLED", True)
    monkeypatch.setattr(settings, "INTENT_CONFIDENCE_THRESHOLD", 0.65)
    monkeypatch.setattr(settings, "SALES_QUERY_ENABLED", True)
    monkeypatch.setattr(settings, "COMMERCE_ASSISTANT_ENABLED", True)

    def attach(chain, route="knowledge_qa", *, confidence=0.95, error=None):
        calls = []
        class RouterModel:
            def with_structured_output(self, schema):
                assert schema is IntentPrediction
                def classify(messages, config=None):
                    calls.append(messages)
                    if error:
                        raise error
                    return IntentPrediction(route=route, intent="通用问答", confidence=confidence,
                        need_clarification=False)
                return RunnableLambda(classify)
        chain.intent_router = IntentRouter(RouterModel())
        return calls
    return attach


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(("route", "service", "intent", "response_type", "question"), [
    ("sales_query", "SalesQueryService", "销量查询", "data_query", "上个月谁卖得最好？"),
    ("commerce_analysis", "CommerceAssistantService", "经营分析", "analysis_report",
        "帮我完整复盘 XX 店铺最新一场直播的数据，看看整体卖得怎么样"),
])
def test_model_business_routes_reach_handler_and_save_once(model_router, monkeypatch,
        streaming, route, service, intent, response_type, question):
    import core.rag_chain as module
    chain = make_chain()
    model_calls = model_router(chain, route)
    handler_calls = []

    class Handler:
        def __init__(self, *args, **kwargs):
            pass
        def events(self, question, history, **kwargs):
            handler_calls.append(question)
            yield {"stage": "正在读取业务数据"}
            yield {"answer": "业务处理结果", "context": [], "evidence_sufficient": True,
                "response_type": response_type, "citation_status": "database",
                "query_details": {"status": "succeeded"}}

    def unexpected_knowledge(*args, **kwargs):
        pytest.fail("已确定的业务请求不应进入普通知识问答")

    monkeypatch.setattr(module, service, Handler)
    monkeypatch.setattr(chain, "_create_rag_chain", unexpected_knowledge)
    chain._rag_chain = RunnableLambda(unexpected_knowledge)
    if streaming:
        events = list(chain.stream_answer(question))
        assert events[-1]["event"] == "done"
        result = events[-1]["data"]
    else:
        result = chain.ask(question)

    assert len(model_calls) == 1
    assert handler_calls == [question]
    assert result["intent"] == intent
    assert result["response_type"] == response_type
    assert result["answer"] == "业务处理结果"
    with database.session_scope() as db:
        row = db.query(QAInteraction).one()
        assert row.status == "completed" and row.intent == intent


@pytest.mark.parametrize("streaming", [False, True])
def test_router_failure_reports_retry_and_never_starts_business(model_router, monkeypatch, streaming):
    chain = make_chain()
    model_router(chain, error=TimeoutError("private-provider-detail"))
    def unexpected_handler(*args, **kwargs):
        pytest.fail("分类失败后不能猜测业务并继续执行")
    monkeypatch.setattr(chain, "_invoke_route", unexpected_handler)
    monkeypatch.setattr(chain, "_route_stream", unexpected_handler)

    if streaming:
        events = list(chain.stream_answer("直播和视频总销量前五主播"))
        assert events[-1]["event"] == "error"
        assert not any(event["event"] in {"delta", "done"} for event in events)
        result = events[-1]["data"]
    else:
        result = chain.ask("直播和视频总销量前五主播")

    assert result["status"] == "failed"
    assert result["response_type"] == "error"
    assert "稍后重试" in result["answer"]
    assert "private" not in json.dumps(result)
    with database.session_scope() as db:
        assert db.query(QAInteraction).one().status == "failed"
    assert chain.get_chat_history() == []


@pytest.mark.parametrize("streaming", [False, True])
def test_disabled_model_route_reports_feature_limit_without_execution(model_router, monkeypatch, streaming):
    chain = make_chain()
    model_router(chain, "commerce_analysis")
    monkeypatch.setattr(settings, "COMMERCE_ASSISTANT_ENABLED", False)
    def unexpected_handler(*args, **kwargs):
        pytest.fail("已关闭的业务不能执行，也不能改走知识问答")
    monkeypatch.setattr(chain, "_invoke_route", unexpected_handler)
    monkeypatch.setattr(chain, "_route_stream", unexpected_handler)

    result = list(chain.stream_answer("帮我复盘店铺表现"))[-1]["data"] if streaming else chain.ask("帮我复盘店铺表现")

    assert result["response_type"] == "error"
    assert "经营分析功能当前未开启" in result["answer"]


@pytest.mark.parametrize("streaming", [False, True])
def test_uncertain_model_route_clarifies_without_executing(model_router, monkeypatch, streaming):
    chain = make_chain()
    model_router(chain, "sales_query", confidence=0.2)
    def unexpected_handler(*args, **kwargs):
        pytest.fail("不确定的路由只能追问，不能查询或生成业务答案")
    monkeypatch.setattr(chain, "_invoke_route", unexpected_handler)
    monkeypatch.setattr(chain, "_route_stream", unexpected_handler)

    if streaming:
        events = list(chain.stream_answer("看看这个怎么样"))
        result = events[-1]["data"]
        assert "".join(event["data"]["text"] for event in events if event["event"] == "delta") == result["answer"]
    else:
        result = chain.ask("看看这个怎么样")
    assert result["response_type"] == "clarify"
    assert result["intent"] == "待澄清"
    assert result["answer"]


@pytest.mark.parametrize("streaming", [False, True])
def test_model_knowledge_route_preserves_normal_answer_flow(model_router, streaming):
    chain = make_chain()
    calls = model_router(chain)
    result = list(chain.stream_answer("什么是直播销量？"))[-1]["data"] if streaming else chain.ask("什么是直播销量？")
    assert len(calls) == 1
    assert result["answer"] == "第一段。第二段。"
    assert result["sources"][0]["source"] == "policy.txt"
    with database.session_scope() as db:
        assert db.query(QAInteraction).one().status == "completed"


@pytest.mark.parametrize("partial", [False, True])
def test_model_error_never_completes_partial_answer(isolated, partial):
    class FailingModel(FakeListChatModel):
        def _stream(self, *args, **kwargs):
            if partial:
                yield ChatGenerationChunk(message=AIMessageChunk(content="半段答案"))
            raise RuntimeError("private-provider-detail")
    before = first_count()
    chain = make_chain(FailingModel(responses=["unused"]))
    events = list(chain.stream_answer("如何报销"))
    assert events[-1]["event"] == "error" and not any(event["event"] == "done" for event in events)
    assert "private-provider-detail" not in json.dumps(events)
    assert first_count() == before + int(partial)
    with database.session_scope() as db:
        assert db.query(QAInteraction).one().status == "failed"
    assert chain.get_chat_history() == []


def test_closing_stream_does_not_save_partial_answer(isolated):
    chain = make_chain()
    stream = chain.stream_answer("如何报销")
    next(event for event in stream if event["event"] == "delta")
    stream.close()
    with database.session_scope() as db:
        assert db.query(QAInteraction).count() == 0


def test_original_complete_interface_is_preserved_and_not_counted_as_first_token(isolated):
    before = first_count()
    api = TestClient(make_app(make_chain()))
    response = api.post("/qa/ask", json={"question": "如何报销"})
    assert response.status_code == 200 and response.json()["answer"] == "第一段。第二段。"
    assert response.json()["interaction_id"]
    assert first_count() == before


def test_stream_route_validates_question_and_auth_before_sending_events(isolated, monkeypatch):
    api = TestClient(make_app(make_chain()))
    assert api.post("/qa/ask/stream", json={"question": "  "}).status_code == 400
    assert api.post("/qa/ask/stream", json={"question": "如何报销"}, headers={"Authorization": "Bearer invalid"}).status_code == 401
    monkeypatch.setattr(settings, "AUTH_REQUIRED", True)
    assert api.post("/qa/ask/stream", json={"question": "如何报销"}).status_code == 401
    headers = {"Authorization": "Bearer " + create_access_token("alice", tenant_id="company-a")}
    response = api.post("/qa/ask/stream", json={"question": "如何报销"}, headers=headers)
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-accel-buffering"] == "no"
    assert response.headers["cache-control"] == "no-cache"
    with database.session_scope() as db:
        row = db.query(QAInteraction).one()
        assert row.username == "alice" and row.tenant_id == "company-a"


class StreamResponse:
    def __init__(self, events, status_code=200):
        self.status_code = status_code
        self.text = "request error"
        self.closed = False
        self.events = events
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.closed = True
    def iter_lines(self, **kwargs):
        for event in self.events:
            yield "event: " + event["event"]
            yield "data: " + json.dumps(event["data"], ensure_ascii=False)
            yield ""


def test_sse_parser_handles_chinese_multiline_data_and_heartbeats():
    response = SimpleNamespace(iter_lines=lambda **kwargs: iter([
        ": ping", "", "event: delta", "data: {", 'data: "text": "中文\\n换行"', "data: }", "",
    ]))
    assert list(read_answer_events(response)) == [{"event": "delta", "data": {"text": "中文\n换行"}}]
    assert response.encoding == "utf-8"


def test_sse_parser_rejects_incomplete_event():
    response = SimpleNamespace(iter_lines=lambda **kwargs: iter(["event: delta", 'data: {"text":"partial"}']))
    with pytest.raises(ValueError, match="未接收完整"):
        list(read_answer_events(response))


@pytest.mark.parametrize("ending", ["done", "error", "disconnected", "not_loaded"])
def test_page_streams_once_and_only_saves_complete_results(ending):
    from streamlit.testing.v1 import AppTest
    from streamlit.testing.v1.element_tree import TESTING_KEY
    events = [{"event": "status", "data": {"message": "正在检索"}},
        {"event": "delta", "data": {"text": "第一段", "first_token_ms": 1250.0}},
        {"event": "delta", "data": {"text": "第二段", "first_token_ms": 1250.0}}]
    if ending == "done":
        events.append({"event": "done", "data": {"answer": "第一段第二段", "sources": [{"source": "policy.txt", "content": "原文"}],
            "interaction_id": "interaction", "trace_id": "trace", "first_token_ms": 1250.0}})
    elif ending == "error":
        events.append({"event": "error", "data": {"answer": "生成失败"}})
    streamed = StreamResponse(events, status_code=404 if ending == "not_loaded" else 200)
    def request(method, url, **kwargs):
        if method == "POST":
            assert url.endswith("/qa/ask/stream") and kwargs["stream"] is True
            return streamed
        return SimpleNamespace(status_code=200, ok=True, json=lambda: {"items": [], "history": []})
    page = next((Path(__file__).resolve().parents[2] / "frontend/pages").glob("1_*.py"))
    with patch("frontend.api_client.transport.request", side_effect=request) as transport:
        app = AppTest.from_file(str(page)).run(timeout=20)
        # AppTest 在脚本线程外调用格式函数，使用快照避免读取不具备上下文的 st.session_state。
        session_names = dict(app.session_state["sessions"])
        app.session_state[TESTING_KEY][app.selectbox[0].id] = session_names.__getitem__
        app.chat_input[0].set_value("如何报销").run(timeout=20)
        assert not list(app.exception)
        submitted = [call for call in transport.call_args_list if call.args[0] == "POST"]
        assert len(submitted) == 1 and streamed.closed
        messages = app.session_state["messages"]
        if ending == "done":
            assert messages[-1]["content"] == "第一段第二段" and messages[-1]["interaction_id"] == "interaction"
            assert any("首字耗时：1.25 秒" in caption.value for caption in app.caption)
            assert not list(app.error)
        else:
            assert all(message["role"] != "assistant" for message in messages)
            expected = "重启后端" if ending == "not_loaded" else "回答未完成"
            assert any(expected in error.value for error in app.error)


@contextmanager
def live_server(app):
    # 使用随机本地端口，完整走 HTTP 验证首段提前到达；TestClient 会缓冲整个响应。
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()


def test_real_http_first_chunk_arrives_before_completion_and_feedback_still_works(isolated):
    release = threading.Event()
    class PausingModel(FakeListChatModel):
        def _stream(self, *args, **kwargs):
            yield ChatGenerationChunk(message=AIMessageChunk(content="第一段"))
            assert release.wait(timeout=5)
            yield ChatGenerationChunk(message=AIMessageChunk(content="第二段",
                usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}))
    chain = make_chain(PausingModel(responses=["unused"]))
    try:
        with live_server(make_app(chain)) as base:
            with requests.post(base + "/qa/ask/stream", json={"question": "如何报销"}, stream=True, timeout=5) as response:
                events = read_answer_events(response)
                first = next(event for event in events if event["event"] == "delta")
                assert first["data"]["text"] == "第一段" and not release.is_set()
                assert requests.get(base + "/ping", timeout=1).json() == {"ok": True}
                with database.session_scope() as db:
                    assert db.query(QAInteraction).count() == 0
                release.set()
                done = list(events)[-1]["data"]
                assert done["answer"] == "第一段第二段"
                assert done["first_token_ms"] == first["data"]["first_token_ms"]
            feedback = requests.post(base + "/qa/feedback", json={"interaction_id": done["interaction_id"], "rating": "positive"}, timeout=5)
            assert feedback.status_code == 200
            with database.session_scope() as db:
                row = db.query(QAInteraction).one()
                assert row.input_tokens == 10 and row.output_tokens == 2
    finally:
        release.set()


def test_real_http_disconnect_closes_model_and_does_not_save_partial_answer(isolated, timing_spans):
    release, closed = threading.Event(), threading.Event()
    holder = {}
    class PausingModel(FakeListChatModel):
        def _stream(self, *args, **kwargs):
            yield ChatGenerationChunk(message=AIMessageChunk(content="半段答案"))
            release.wait(timeout=5)
            yield ChatGenerationChunk(message=AIMessageChunk(content="剩余答案"))
    chain = make_chain(PausingModel(responses=["unused"]))
    original = chain.stream_answer
    def capture(*args, **kwargs):
        holder["cancelled"] = kwargs["cancelled"]
        try:
            yield from original(*args, **kwargs)
        finally:
            closed.set()
    chain.stream_answer = capture
    try:
        with live_server(make_app(chain)) as base:
            with requests.post(base + "/qa/ask/stream", json={"question": "如何报销"}, stream=True, timeout=5) as response:
                next(event for event in read_answer_events(response) if event["event"] == "delta")
            assert holder["cancelled"].wait(timeout=2)
            release.set()
            assert closed.wait(timeout=2)
            spans = {item.name: item for item in timing_spans.get_finished_spans()}
            assert spans["rag.generation"].attributes["rag.cancelled"] is True
            assert spans["rag.generation.output"].end_time is not None
            with database.session_scope() as db:
                assert db.query(QAInteraction).count() == 0
    finally:
        release.set()


@pytest.mark.parametrize("provider_name", ["openai", "siliconflow"])
def test_openai_compatible_stream_preserves_chinese_and_reported_usage(isolated, monkeypatch, provider_name):
    from fastapi.responses import StreamingResponse
    from core.llm_client import LLMClient
    provider = FastAPI()
    @provider.post("/v1/chat/completions")
    def completion(payload: dict):
        assert payload["stream"] is True
        assert payload["stream_options"]["include_usage"] is True
        def chunks():
            for text in ("中文", "\n答案"):
                chunk = {"id": "mock", "object": "chat.completion.chunk", "created": 0, "model": "mock",
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]}
                yield "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"
            usage = {"id": "mock", "object": "chat.completion.chunk", "created": 0, "model": "mock", "choices": [],
                "usage": {"prompt_tokens": 11, "completion_tokens": 6, "total_tokens": 17}}
            yield "data: " + json.dumps(usage) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(chunks(), media_type="text/event-stream")
    with live_server(provider) as base:
        monkeypatch.setattr(settings, "LLM_PROVIDER", provider_name)
        prefix = provider_name.upper()
        monkeypatch.setattr(settings, prefix + "_API_KEY", "test-key")
        monkeypatch.setattr(settings, prefix + "_BASE_URL", base + "/v1")
        events = list(make_chain(LLMClient().get_llm()).stream_answer("如何报销"))
        assert events[-1]["data"]["answer"] == "中文\n答案"
        assert len([event for event in events if event["event"] == "delta"]) == 2
        with database.session_scope() as db:
            row = db.query(QAInteraction).one()
            assert row.input_tokens == 11 and row.output_tokens == 6


def test_ollama_stream_preserves_chinese_and_first_token_timing(isolated, monkeypatch):
    from fastapi.responses import StreamingResponse
    from core.llm_client import LLMClient
    provider = FastAPI()
    @provider.post("/api/chat")
    def completion(payload: dict):
        assert payload.get("stream", True) is True  # Ollama 未指定时默认以 NDJSON 流式返回。
        def chunks():
            for text in ("中文", "答案"):
                yield json.dumps({"model": "mock", "message": {"role": "assistant", "content": text}, "done": False}, ensure_ascii=False) + "\n"
            yield json.dumps({"model": "mock", "message": {"role": "assistant", "content": ""}, "done": True,
                "prompt_eval_count": 10, "eval_count": 2}) + "\n"
        return StreamingResponse(chunks(), media_type="application/x-ndjson")
    with live_server(provider) as base:
        monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")
        monkeypatch.setattr(settings, "OLLAMA_BASE_URL", base)
        events = list(make_chain(LLMClient().get_llm()).stream_answer("如何报销"))
        assert events[-1]["data"]["answer"] == "中文答案"
        assert events[-1]["data"]["first_token_ms"] >= 0


def test_stream_does_not_report_done_when_database_save_fails(isolated, timing_spans):
    from sqlalchemy import event
    def refuse_completed(session, *args):
        if any(isinstance(row, QAInteraction) and row.status == "completed" for row in session.new):
            raise RuntimeError("save failed")
    event.listen(database.SessionLocal, "before_flush", refuse_completed)
    try:
        events = list(make_chain().stream_answer("如何报销"))
        assert events[-1]["event"] == "error"
        assert not any(item["event"] == "done" for item in events)
        from opentelemetry.trace import StatusCode
        persistence = [item for item in timing_spans.get_finished_spans() if item.name == "rag.persist"]
        assert persistence[0].status.status_code == StatusCode.ERROR
        assert persistence[0].attributes["error.type"] == "RuntimeError"
        assert all(item.end_time is not None for item in persistence)
        with database.session_scope() as db:
            assert db.query(QAInteraction).one().status == "failed"
    finally:
        event.remove(database.SessionLocal, "before_flush", refuse_completed)


def test_langfuse_finish_contains_first_token_timing_without_content(isolated, monkeypatch):
    from core.langfuse_tracker import LangfuseTracker
    monkeypatch.setattr(settings, "LANGFUSE_CAPTURE_CONTENT", False)
    tracker = LangfuseTracker("session")
    endings, updates = [], []
    tracker._trace = SimpleNamespace(
        generation=lambda **kwargs: SimpleNamespace(end=lambda **kwargs: endings.append(kwargs)),
        update=lambda **kwargs: updates.append(kwargs),
    )
    tracker._client = SimpleNamespace(flush=lambda: None)
    tracker.finish("private question", "private answer", [], "mock", first_token_ms=1250.0)
    assert endings[0]["metadata"]["first_token_ms"] == 1250.0
    assert updates[0]["metadata"]["first_token_ms"] == 1250.0
    assert "private" not in json.dumps(endings + updates)


@pytest.fixture
def timing_spans(monkeypatch):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from core import telemetry
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("qa-timing-test")
    monkeypatch.setattr(telemetry.trace, "get_tracer", lambda *args, **kwargs: tracer)
    yield exporter
    provider.shutdown()


def test_timing_splits_first_visible_text_and_excludes_consumer_wait(isolated, monkeypatch, timing_spans):
    from core import rag_chain as module
    from core.telemetry import span
    clock = [0.0]
    class TimedModel(FakeListChatModel):
        def _stream(self, *args, **kwargs):
            clock[0] += 2
            yield ChatGenerationChunk(message=AIMessageChunk(content=" "))
            clock[0] += 3
            yield ChatGenerationChunk(message=AIMessageChunk(content="正文"))
            clock[0] += 4
            yield ChatGenerationChunk(message=AIMessageChunk(content="结尾"))
    monkeypatch.setattr(module, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    stream = make_chain(TimedModel(responses=["unused"])).stream_answer("如何报销", started_at=0)
    with span("test.request") as request:
        first = next(event for event in stream if event["event"] == "delta")
        assert first["data"]["text"] == " 正文"
        assert first["data"]["first_token_ms"] == 5000
        finished = {item.name: item for item in timing_spans.get_finished_spans()}
        assert "rag.generation.wait_first_chunk" in finished
        assert "rag.generation.output" not in finished
        clock[0] += 10  # 模拟下游处理或客户端发送等待，不能算成上游生成耗时。
        rest = list(stream)
        request_id = request.get_span_context().span_id
    spans = {item.name: item for item in timing_spans.get_finished_spans()}
    assert rest[-1]["data"]["answer"] == " 正文结尾"
    assert spans["test.request"].attributes["rag.first_token_ms"] == 5000
    assert "rag.first_token_ms" not in spans["rag.generation.output"].attributes
    output = spans["rag.generation.output"]
    assert output.attributes["rag.stream.consumer_wait_ms"] == 10000
    assert output.attributes["rag.generation.upstream_ms"] == 4000
    assert output.parent.span_id == spans["rag.generation"].context.span_id
    assert spans["rag.generation"].parent.span_id == request_id
    assert spans["rag.generation.wait_first_chunk"].end_time <= output.start_time
    assert spans["rag.generation"].end_time <= spans["rag.persist"].start_time
    assert {"rag.intent", "rag.history", "rag.prepare", "rag.retrieval", "rag.persist", "rag.finalize"} <= spans.keys()
    assert not any(item.events for item in spans.values())
    assert "如何报销" not in json.dumps([dict(item.attributes) for item in spans.values()], ensure_ascii=False)


@pytest.mark.parametrize("has_history", [False, True])
def test_timing_records_rewrite_only_when_used(isolated, timing_spans, has_history):
    from core.telemetry import span
    model = FakeListChatModel(responses=["独立问题", "最终回答"] if has_history else ["最终回答"])
    chain = make_chain(model)
    if has_history:
        with database.session_scope() as db:
            db.add(QAInteraction(session_id="session", username="anonymous", question="报销", answer="上一轮", status="completed"))
    with span("test.request"):
        events = list(chain.stream_answer("具体怎么申请", "session"))
    spans = {item.name: item for item in timing_spans.get_finished_spans()}
    assert events[-1]["data"]["answer"] == "最终回答"
    assert ("rag.rewrite" in spans) is has_history
    if has_history:
        assert spans["rag.rewrite"].parent.span_id == spans["rag.retrieval"].context.span_id


@pytest.mark.parametrize("partial", [False, True])
def test_timing_model_failure_ends_active_phase_without_exception_content(isolated, timing_spans, partial):
    from core.telemetry import span
    from opentelemetry.trace import StatusCode
    class FailingModel(FakeListChatModel):
        def _stream(self, *args, **kwargs):
            if partial:
                yield ChatGenerationChunk(message=AIMessageChunk(content="半段答案"))
            raise RuntimeError("private-provider-detail")
    with span("test.request"):
        events = list(make_chain(FailingModel(responses=["unused"])).stream_answer("如何报销"))
    spans = {item.name: item for item in timing_spans.get_finished_spans()}
    phase = "rag.generation.output" if partial else "rag.generation.wait_first_chunk"
    assert events[-1]["event"] == "error"
    for name in (phase, "rag.generation", "test.request"):
        assert spans[name].status.status_code == StatusCode.ERROR
        assert spans[name].attributes["error.type"] == "RuntimeError"
        assert spans[name].end_time is not None
        assert not spans[name].events
    assert "private-provider-detail" not in json.dumps([dict(item.attributes) for item in spans.values()])


def test_timing_cancel_after_first_text_closes_spans_and_preserves_context(isolated, timing_spans):
    from core.telemetry import span
    from opentelemetry import trace
    stream = make_chain().stream_answer("如何报销")
    with span("test.request") as request:
        next(event for event in stream if event["event"] == "delta")
        stream.close()
        assert trace.get_current_span().get_span_context().span_id == request.get_span_context().span_id
    spans = {item.name: item for item in timing_spans.get_finished_spans()}
    assert spans["rag.generation"].attributes["rag.cancelled"] is True
    assert spans["test.request"].attributes["rag.cancelled"] is True
    assert spans["rag.generation.output"].end_time is not None
    with database.session_scope() as db:
        assert db.query(QAInteraction).count() == 0


def test_timing_cancel_before_visible_text_closes_wait_phase(isolated, timing_spans):
    from core.telemetry import span
    from opentelemetry import trace
    def text_chunks(inputs, config):
        yield " "
        yield "正文"
    chain = make_chain()
    with span("test.request") as request:
        stream = chain._stream_generation(RunnableLambda(text_chunks), {}, {})
        assert next(stream) == {"answer": " "}
        stream.close()
        assert trace.get_current_span().get_span_context().span_id == request.get_span_context().span_id
    spans = {item.name: item for item in timing_spans.get_finished_spans()}
    assert spans["rag.generation"].attributes["rag.cancelled"] is True
    assert spans["rag.generation.wait_first_chunk"].end_time is not None
    assert "rag.generation.output" not in spans


def test_timing_complete_answer_has_generation_without_stream_phases(isolated, timing_spans):
    from core.telemetry import span
    with span("test.request"):
        result = make_chain().ask("如何报销", persist=False)
    spans = {item.name: item for item in timing_spans.get_finished_spans()}
    assert result["status"] == "completed"
    assert {"rag.retrieval", "rag.generation", "rag.finalize"} <= spans.keys()
    assert not any(name.startswith("rag.generation.") for name in spans)
    assert "rag.persist" not in spans


def test_timing_no_evidence_does_not_report_model_generation(isolated, timing_spans):
    from core.telemetry import span
    with span("test.request"):
        events = list(make_chain(documents=[]).stream_answer("如何报销"))
    spans = {item.name: item for item in timing_spans.get_finished_spans()}
    assert events[-1]["data"]["response_type"] == "refuse"
    assert "rag.retrieval" in spans
    assert "rag.generation" not in spans


def test_timing_stream_route_keeps_all_stages_on_one_trace(isolated, timing_spans):
    response = TestClient(make_app(make_chain())).post("/qa/ask/stream", json={"question": "如何报销"})
    assert response.status_code == 200
    spans = {item.name: item for item in timing_spans.get_finished_spans()}
    request = spans["rag.ask.stream"]
    assert request.attributes["rag.first_token_ms"] >= 0
    assert spans["rag.generation"].parent.span_id == request.context.span_id
    assert all(item.context.trace_id == request.context.trace_id for item in spans.values())


@pytest.mark.parametrize("provider_name", ["openai", "siliconflow"])
def test_http_timing_covers_rewrite_and_stream_with_preserved_usage(isolated, monkeypatch, timing_spans, provider_name):
    from fastapi.responses import StreamingResponse
    from core.llm_client import LLMClient
    from core.telemetry import span
    provider = FastAPI()
    calls = []
    @provider.post("/v1/chat/completions")
    def completion(payload: dict):
        calls.append(payload)
        if not payload.get("stream"):
            return {"id": "rewrite", "object": "chat.completion", "created": 0, "model": "mock",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "差旅报销材料"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}
        assert payload["stream_options"]["include_usage"] is True
        def chunks():
            yield ": heartbeat\n\n"
            for text in ("", " ", "中文", "答案"):
                chunk = {"id": "answer", "object": "chat.completion.chunk", "created": 0, "model": "mock",
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]}
                yield "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"
            yield "data: " + json.dumps({"id": "answer", "object": "chat.completion.chunk", "created": 0, "model": "mock",
                "choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 6, "total_tokens": 17}}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(chunks(), media_type="text/event-stream")
    with database.session_scope() as db:
        db.add(QAInteraction(session_id="session", username="anonymous", question="如何报销", answer="上一轮", status="completed"))
    monkeypatch.setattr(settings, "OTEL_ENABLED", True)
    with live_server(provider) as base:
        monkeypatch.setattr(settings, "LLM_PROVIDER", provider_name)
        prefix = provider_name.upper()
        monkeypatch.setattr(settings, prefix + "_API_KEY", "test-key")
        monkeypatch.setattr(settings, prefix + "_BASE_URL", base + "/v1")
        model = LLMClient().get_llm()
        retrieved = []
        try:
            with span("test.request"):
                events = list(make_chain(model, retrieved=retrieved).stream_answer("需要哪些材料", "session"))
        finally:
            model.http_client.close()
            import asyncio
            asyncio.run(model.http_async_client.aclose())
    assert events[-1]["data"]["answer"] == " 中文答案"
    assert retrieved == ["差旅报销材料"]
    assert len(calls) == 2 and not calls[0].get("stream") and calls[1]["stream"]
    finished = timing_spans.get_finished_spans()
    spans = {item.name: item for item in finished}
    requests = [item for item in finished if item.name == "llm.http.request"]
    assert len(requests) == 2
    assert requests[0].parent.span_id == spans["rag.rewrite"].context.span_id
    assert requests[1].parent.span_id == spans["rag.generation"].context.span_id
    assert all(item.context.trace_id == spans["test.request"].context.trace_id for item in finished)
    assert "llm.http.first_event_received" not in requests[0].attributes
    assert requests[1].attributes["llm.http.first_event_received"] is True
    assert all(item.attributes["llm.http.response_complete"] is True for item in requests)
    assert spans["llm.http.wait_first_event"].end_time <= spans["rag.generation.wait_first_chunk"].end_time
    assert spans["llm.http.request"].end_time <= spans["rag.generation"].end_time
    with database.session_scope() as db:
        row = db.query(QAInteraction).filter(QAInteraction.question == "需要哪些材料").one()
        assert row.input_tokens == 18 and row.output_tokens == 9
