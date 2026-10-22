"""Agent integration checks with scripted decisions and real isolated business queries."""
from datetime import date, datetime, timedelta, timezone
import inspect
import json
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from config.settings import settings
from core import agent_tools, database
from core.auth import ANONYMOUS_USER, AuthUser
from core.business_seed import seed_business_data
from core.conversation_agent import ConversationAgent, CellRef, permission_stamp
from core.db_models import QAInteraction
from core.memory_manager import MemoryManager
from core.rag_chain import RAGChain
from tests.问答功能.test_qa_streaming import isolated, make_app


@pytest.fixture
def agent_env(isolated, monkeypatch):
    monkeypatch.setattr(settings, "AGENT_ENABLED", True)
    monkeypatch.setattr(settings, "AGENT_MAX_ROUNDS", 16)
    monkeypatch.setattr(settings, "AGENT_MAX_TOOL_CALLS", 24)
    monkeypatch.setattr(settings, "SALES_QUERY_ENABLED", True)
    seed_business_data(database.engine, date(2026, 10, 20), days=8)


def call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": name + "_call"}])


def output(messages):
    return json.loads(next(m.content for m in reversed(messages) if isinstance(m, ToolMessage)))


class ScriptedModel:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.messages, self.offered = [], []

    def bind_tools(self, tools):
        self.offered.append({t.name for t in tools})
        return self

    def invoke(self, messages, **kwargs):
        self.messages.append(list(messages))
        reply = next(self.responses)
        return reply(messages) if callable(reply) else reply


def chain_for(model, retriever=None):
    chain = object.__new__(RAGChain)
    chain.llm, chain.memory_manager = model, MemoryManager()
    chain.retriever = retriever or SimpleNamespace(retrieve=lambda *args, **kwargs: [])
    # Any accidental return to preliminary routing must fail this integration test.
    chain.intent_router = SimpleNamespace(classify=lambda *a, **k: pytest.fail("Agent must not classify first"))
    return chain


def finish(answer="已完成", **kwargs):
    return call("finish_answer", answer=answer, **kwargs)


def shop_script():
    return [
        call("discover_capabilities", capabilities=["business"]),
        call("read_business_schema", tables=["shop", "jl_user", "live_base", "live_sales_info"]),
        call("query_database", tables=["shop"], sql="SELECT shop_id,name FROM shop WHERE name LIKE '%清研美妆%'"),
        call("query_database", tables=["jl_user", "live_base"], sql="SELECT l.id AS live_id,l.live_create_time FROM live_base l "
            "JOIN jl_user u ON u.anchor_id=l.anchor_id WHERE u.shop_id=101 AND l.finish_time>0 AND l.is_show=0 "
            "ORDER BY l.live_create_time DESC,l.id DESC LIMIT 1"),
        lambda m: finish("已确认本次直播，当前是本地模拟数据。", goal="复盘清研美妆最新一场直播", facts={
            "active_live_id": {"evidence_id": output(m)["evidence_id"], "row": 0, "column": "live_id"}}),
    ]


@pytest.mark.parametrize("streaming", [False, True])
def test_default_entry_can_answer_without_tools_or_router(agent_env, streaming):
    chain = chain_for(ScriptedModel(AIMessage(content="GMV指商品交易总额。")))
    result = list(chain.stream_answer("GMV是什么意思？", "general"))[-1]["data"] if streaming else chain.ask("GMV是什么意思？", "general")
    assert result["answer"] == "GMV指商品交易总额。"
    assert result["query_details"]["steps"] == []
    assert result["query_details"]["analysis_type"] == "agent"
    assert result["evidence_sufficient"]
    with database.session_scope() as db:
        assert db.query(QAInteraction).count() == 1


def test_two_turns_reuse_pinned_session_and_execute_comparison(agent_env):
    first_model = ScriptedModel(*shop_script())
    chain = chain_for(first_model)
    first = chain.ask("复盘清研美妆最新一场直播", "shop")
    state = first["query_details"]["task_state"]
    pinned = state["facts"]["active_live_id"]
    assert pinned["value"]
    baseline = {}

    def compare(m):
        baseline.update(output(m))
        assert baseline["rows"][0][0] != pinned["value"]
        return finish("已按刚才确认的场次找到上一场，可据此继续对比。", facts={
            "previous_live_id": {"evidence_id": baseline["evidence_id"], "row": 0, "column": "live_id"}})

    second_model = ScriptedModel(
        call("discover_capabilities", capabilities=["business"]),
        call("read_business_schema", tables=["live_base", "jl_user"]),
        call("query_database", tables=["live_base", "jl_user"], sql=f"SELECT l.id AS live_id FROM live_base l "
            f"JOIN jl_user u ON u.anchor_id=l.anchor_id WHERE u.shop_id=101 AND l.id<>{pinned['value']} "
            "AND l.finish_time>0 AND l.is_show=0 ORDER BY l.live_create_time DESC,l.id DESC LIMIT 1"), compare)
    chain.llm = second_model
    second = list(chain.stream_answer("那和上一场比呢？", "shop"))[-1]["data"]
    second_state = second["query_details"]["task_state"]
    assert second_state["facts"]["active_live_id"] == pinned
    assert second_state["facts"]["previous_live_id"]["value"] != pinned["value"]
    assert str(pinned["value"]) in second_model.messages[0][0].content
    assert first_model.offered[0] == {"discover_capabilities", "finish_answer"}
    assert "query_database" in first_model.offered[1]


def test_schema_is_required_and_sql_write_is_rejected(agent_env):
    model = ScriptedModel(call("discover_capabilities", capabilities=["business"]),
        call("query_database", tables=["shop"], sql="SELECT shop_id FROM shop"),
        call("read_business_schema", tables=["shop"]),
        call("query_database", tables=["shop"], sql="DELETE FROM shop"),
        finish("查询被检查拦截，请调整查询。", response_type="clarify"))
    result = chain_for(model).ask("查店铺", "safe")
    failed = [s for s in result["query_details"]["steps"] if s["status"] == "failed"]
    assert len(failed) == 2
    assert "说明" in failed[0]["summary"] and "SELECT" in failed[1]["summary"]
    assert not result["query_details"]["queries"]


def test_query_then_knowledge_and_calculation_can_share_one_loop(agent_env):
    received = []
    retriever = SimpleNamespace(retrieve=lambda query, **kwargs: received.append(kwargs) or [
        Document(page_content="先比较同口径的销售额，再分析结构变化。", metadata={"source": "复盘规范", "document_id": "d1"})])

    def calc(m):
        ref = output(m)["evidence_id"]
        return call("calculate", operation="percent_change", references=[
            {"evidence_id": ref, "column": "current", "row": 0}, {"evidence_id": ref, "column": "baseline", "row": 0}])

    model = ScriptedModel(call("discover_capabilities", capabilities=["business", "calculation"]),
        call("read_business_schema", tables=["shop"]),
        call("query_database", tables=["shop"], sql="SELECT 80 AS current,100 AS baseline FROM shop LIMIT 1"),
        calc, call("discover_capabilities", capabilities=["knowledge"]),
        call("search_knowledge", query="公司复盘规范"), finish("模拟数据下降20%，按内部规范比较[1]。"))
    result = chain_for(model, retriever).ask("查数、比较，再按内部规范解释", "mixed")
    assert received[0]["user"] == ANONYMOUS_USER
    assert result["sources"][0]["citation_id"] == 1
    assert "search_knowledge" in result["query_details"]["tools_used"]
    calculations = [v for v in result["query_details"]["task_state"]["evidence"].values() if v["kind"] == "calculation"]
    assert calculations[0]["rows"] == [[-20]]


def test_reset_task_discards_old_conditions_and_results(agent_env):
    chain = chain_for(ScriptedModel(*shop_script()))
    old = chain.ask("复盘清研美妆最新一场", "switch")["query_details"]["task_state"]
    chain.llm = ScriptedModel(call("discover_capabilities", capabilities=["business"]),
        call("read_business_schema", tables=["shop"]),
        call("query_database", tables=["shop"], sql="SELECT shop_id FROM shop WHERE shop_id=102"),
        lambda m: finish("已改看另一家店铺近7天，后续需按新条件查询。", goal="店铺102近7天", reset_task=True,
            facts={"shop_id": {"evidence_id": output(m)["evidence_id"], "column": "shop_id", "row": 0}}))
    new = chain.ask("改看店铺102近7天", "switch")["query_details"]["task_state"]
    assert "active_live_id" not in new["facts"]
    assert not set(old["evidence"]) & set(new["evidence"])
    assert new["facts"]["shop_id"]["value"] == 102


def test_invalid_fact_cannot_be_saved_and_repeated_failures_stop(agent_env):
    bad = finish("编造结果", facts={"shop_id": {"evidence_id": "invented", "column": "shop_id"}})
    result = chain_for(ScriptedModel(bad, bad, bad)).ask("查店铺", "bad")
    assert result["query_details"]["stop_reason"] == "repeated_failure"
    assert not result["query_details"]["task_state"]["facts"]
    assert result["response_type"] == "error"


def test_json_protocol_for_model_without_native_tools(agent_env):
    model = SimpleNamespace(invoke=lambda *args, **kwargs: AIMessage(content=json.dumps(
        {"tool": "finish_answer", "arguments": {"answer": "可以直接解释概念。"}}, ensure_ascii=False)))
    result = chain_for(model).ask("解释概念", "json")
    assert result["answer"] == "可以直接解释概念。"


def test_permission_change_removes_saved_evidence_but_age_does_not(agent_env):
    chain = chain_for(ScriptedModel(*shop_script()))
    saved = chain.ask("复盘清研美妆", "expires")["query_details"]["task_state"]
    agent = ConversationAgent(None, None)
    assert agent._restore(saved, AuthUser(username="another"))["evidence"] == {}
    for item in saved["evidence"].values():
        item["checked_at"] = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
    restored = agent._restore(saved, ANONYMOUS_USER)
    assert restored["evidence"] == saved["evidence"]
    assert restored["facts"] == saved["facts"]


def test_clear_and_user_isolation_remove_task_state(agent_env):
    chain = chain_for(ScriptedModel(*shop_script()))
    chain.ask("复盘清研美妆", "private")
    assert chain._persistent_task_state("private", ANONYMOUS_USER)
    assert not chain._persistent_task_state("private", AuthUser(username="other"))
    chain.clear_session("private")
    assert not chain._persistent_task_state("private", ANONYMOUS_USER)


def test_nondefault_tenant_cannot_access_shared_business_database(agent_env):
    user = AuthUser(username="u", tenant_id="another")
    with pytest.raises(ValueError, match="授权"):
        agent_tools.read_schema(["shop"], user)


def test_zero_baseline_is_not_reported_as_zero_change():
    evidence = {"x": {"status": "succeeded", "columns": ["a", "b"], "rows": [[100, 0]]}}
    value = agent_tools.calculate("percent_change", [CellRef(evidence_id="x", column="a"), CellRef(evidence_id="x", column="b")], evidence)
    assert value["rows"] == [[None]] and "基准为零" in value["message"]


@pytest.mark.parametrize("path", ["/qa/ask", "/qa/ask/stream"])
def test_http_endpoints_share_agent_and_persist_once(agent_env, path):
    chain = chain_for(ScriptedModel(finish("直接回答")))
    response = TestClient(make_app(chain)).post(path, json={"question": "你好", "session_id": "http"})
    assert response.status_code == 200
    data = response.json() if path.endswith("/ask") else json.loads(next(
        block.split("data: ", 1)[1] for block in response.text.split("\n\n") if "event: done" in block))
    assert data["query_details"]["analysis_type"] == "agent"
    with database.session_scope() as db:
        assert db.query(QAInteraction).count() == 1


def test_frontend_does_not_render_empty_metric_report_on_failure():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_string("import streamlit as st\nfrom frontend.agent_report import render_agent_report\nrender_agent_report(st.session_state.details)")
    app.session_state.details = {"analysis_type": "agent", "status": "failed", "steps": [
        {"number": 1, "tool": "query_database", "status": "failed", "summary": "字段不存在", "elapsed_ms": 2}], "queries": []}
    app.run()
    assert not app.exception and not app.dataframe
    assert any("字段不存在" in item.value for item in app.caption)


def test_context_retains_general_answer_for_rewriting(agent_env):
    chain = chain_for(ScriptedModel(AIMessage(content="GMV是商品交易总额，通常按某个统计周期计算。")))
    chain.ask("解释GMV", "rewrite")
    model = ScriptedModel(AIMessage(content="GMV就是商品交易总额。"))
    chain.llm = model
    chain.ask("把刚才的回答说简单点", "rewrite")
    assert any("通常按某个统计周期" in m.content for m in model.messages[0] if isinstance(m, AIMessage))


def test_knowledge_without_evidence_refuses_instead_of_inventing_policy(agent_env):
    model = ScriptedModel(call("discover_capabilities", capabilities=["knowledge"]),
        call("search_knowledge", query="公司报销规定"), finish("公司允许随意报销。"))
    result = chain_for(model).ask("公司报销规定是什么？", "no-proof")
    assert result["response_type"] == "refuse"
    assert not result["evidence_sufficient"]
    assert "随意报销" not in result["answer"]


def test_timeout_and_call_limit_stop_without_unbounded_execution(agent_env, monkeypatch):
    monkeypatch.setattr(settings, "AGENT_TIMEOUT_SECONDS", -1)
    result = chain_for(ScriptedModel()).ask("查数据", "timeout")
    assert result["query_details"]["stop_reason"] == "timeout"
    monkeypatch.setattr(settings, "AGENT_TIMEOUT_SECONDS", 180)
    monkeypatch.setattr(settings, "AGENT_MAX_TOOL_CALLS", 1)
    result = chain_for(ScriptedModel(call("discover_capabilities", capabilities=["business"]),
        finish("预算不足，尚未查询业务数据。", response_type="clarify"))).ask("查店铺", "limit")
    assert result["query_details"]["stop_reason"] == "budget_answer"
    assert "达到处理预算" in result["answer"]


def test_concurrent_same_session_request_is_rejected_and_lock_is_released(agent_env):
    from core.rag_chain import _AGENT_SESSION_LOCKS
    from core.memory_manager import session_key
    lock = threading.Lock()
    _AGENT_SESSION_LOCKS[session_key("busy")] = lock
    lock.acquire()
    chain = chain_for(ScriptedModel(AIMessage(content="已恢复正常。")))
    try:
        rejected = chain.ask("你好", "busy", persist=False)
        assert "上一条问题" in rejected["answer"]
    finally:
        lock.release()
    assert chain.ask("你好", "busy", persist=False)["answer"] == "已恢复正常。"


def test_sql_with_percent_literal_uses_mysql_dialect_escaping(agent_env, monkeypatch):
    from sqlalchemy.dialects.mysql.pymysql import MySQLDialect_pymysql
    from sqlalchemy import text
    checked, _ = agent_tools.validate_business_sql("SELECT shop_id FROM shop WHERE name LIKE '%清研%'", agent_tools.query_schema({"shop"}))
    compiled = str(text(checked).compile(dialect=MySQLDialect_pymysql()))
    assert "%%清研%%" in compiled


def test_general_answer_does_not_load_published_knowledge(agent_env, monkeypatch):
    monkeypatch.setattr("core.release_service.resolve_release", lambda *a, **kw: pytest.fail("unused knowledge must not load"))
    result = chain_for(ScriptedModel(AIMessage(content="你好。"))).ask("你好", user=ANONYMOUS_USER)
    assert result["answer"] == "你好。"


def test_knowledge_uses_current_release_manifest_and_actor(agent_env, monkeypatch):
    manifest, captured, marker = {"d": "v2"}, {}, object()
    monkeypatch.setattr("core.release_service.resolve_release", lambda *a, **kw: SimpleNamespace(config={"manifest": manifest}))
    monkeypatch.setattr("core.release_service.release_store", lambda release: marker)
    class Retriever:
        def __init__(self, vector_store):
            assert vector_store is marker
        def retrieve(self, query, **kwargs):
            captured.update(kwargs)
            return [Document(page_content="当前版本规定", metadata={"source": "policy", "document_id": "d"})]
    monkeypatch.setattr("core.retriever.RAGRetriever", Retriever)
    model = ScriptedModel(call("discover_capabilities", capabilities=["knowledge"]),
        call("search_knowledge", query="公司规定"), finish("根据规定[1]。"))
    result = chain_for(model).ask("公司规定是什么", user=ANONYMOUS_USER)
    assert captured["version_manifest"] == manifest and captured["user"] == ANONYMOUS_USER
    assert result["sources"][0]["document_id"] == "d"


def test_quality_guard_still_rejects_unsupported_agent_answer(agent_env, monkeypatch):
    from core.quality_guard import Verdict, REFUSAL
    monkeypatch.setattr(settings, "QUALITY_GUARD_ENABLED", True)
    model = ScriptedModel(call("discover_capabilities", capabilities=["knowledge"]),
        call("search_knowledge", query="公司规定"), finish("没有依据的断言[1]。"))
    model.with_structured_output = lambda schema: SimpleNamespace(invoke=lambda *a, **kw: Verdict(
        safe=True, evidence_sufficient=True, all_claims_supported=False, citations=[1]))
    retriever = SimpleNamespace(retrieve=lambda *a, **kw: [Document(page_content="依据", metadata={"source": "policy"})])
    result = chain_for(model, retriever).ask("按公司规定解释", "guard")
    assert result["answer"] == REFUSAL and result["citation_status"] == "rejected"
    assert not result["evidence_sufficient"]


def test_latest_task_pointer_is_independent_of_second_precision_time(agent_env):
    from datetime import datetime
    chain = chain_for(ScriptedModel(finish("第一轮", goal="第一轮目标"), finish("第二轮", goal="第二轮目标")))
    chain.ask("第一轮", "same-second")
    chain.ask("第二轮", "same-second")
    with database.session_scope() as db:
        for row in db.query(QAInteraction).all():
            row.created_at = datetime(2026, 10, 21, 12)
    assert chain._persistent_task_state("same-second", ANONYMOUS_USER)["goal"] == "第二轮目标"


def test_upgrade_adds_session_pointer_without_losing_clear_marker(tmp_path, monkeypatch):
    from sqlalchemy import create_engine, inspect, text
    engine = create_engine(f"sqlite:///{tmp_path / 'old-session.db'}")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE chat_session_states (id VARCHAR(64) PRIMARY KEY, tenant_id VARCHAR(128), "
            "username VARCHAR(128),session_id VARCHAR(128),cleared_at TIMESTAMP,updated_at TIMESTAMP)"))
        connection.execute(text("INSERT INTO chat_session_states VALUES ('x','default','anonymous','s','2026-10-20',NULL)"))
    monkeypatch.setattr(database, "engine", engine)
    database._ensure_schema_compatibility()
    assert "latest_interaction_id" in {c["name"] for c in inspect(engine).get_columns("chat_session_states")}
    with engine.connect() as connection:
        assert connection.execute(text("SELECT cleared_at FROM chat_session_states")).scalar() == "2026-10-20"
    engine.dispose()


def test_explicit_documented_summary_table_remains_queryable(agent_env):
    result = agent_tools.read_schema(["ads_shop_sale_rank"], ANONYMOUS_USER)
    assert result["tables"][0]["table"] == "ads_shop_sale_rank"
    result = agent_tools.query_business("SELECT shop_id FROM ads_shop_sale_rank LIMIT 2", ["ads_shop_sale_rank"], ANONYMOUS_USER)
    assert result["status"] == "succeeded" and result["row_count"] == 2


def test_private_answer_is_replayed_only_when_permissions_match(agent_env):
    answer = "PRIVATE_OLD_RESULT"
    history = [AIMessage(content="PRIVATE_OLD_RESULT", additional_kwargs={"query_details": {
        "analysis_type": "agent", "private_context": True, "tools_used": [], "used_knowledge": False,
        "task_state": {"permission_stamp": permission_stamp(ANONYMOUS_USER)}}})]
    model = ScriptedModel(AIMessage(content="你好。"))
    list(ConversationAgent(model, None).events("你好", history, user=ANONYMOUS_USER,
        task_state={"permission_stamp": permission_stamp(ANONYMOUS_USER)}))
    assert any(answer in m.content for m in model.messages[0])

    model = ScriptedModel(AIMessage(content="需要重新确认。"))
    list(ConversationAgent(model, None).events("沿用刚才的结果", history,
        user=AuthUser(username="anonymous", role="new-role"), task_state={"permission_stamp": permission_stamp(ANONYMOUS_USER)}))
    assert not any(answer in m.content for m in model.messages[0])


def test_cancelled_agent_request_does_not_save_success(agent_env):
    cancelled = threading.Event()
    cancelled.set()
    chain = chain_for(ScriptedModel())
    list(chain.stream_answer("查数据", "cancel", cancelled=cancelled))
    with database.session_scope() as db:
        assert db.query(QAInteraction).count() == 0


def make_tool_context(user=ANONYMOUS_USER):
    state = {"facts": {}, "evidence": {}}
    return agent_tools.AgentToolContext(user=user, evidence=state["evidence"], documents=[],
        retriever=None, version_manifest=None, capabilities=dict(agent_tools.CAPABILITIES),
        available={"discover_capabilities", "finish_answer"}, current_ids=set(), read_tables=set(),
        task_state=state, question="当前问题")


def test_registered_tools_have_documented_model_parameters_and_real_execution():
    context = make_tool_context()
    tools = agent_tools.build_agent_tools(context)
    assert set(tools) == {"discover_capabilities", "read_business_schema", "query_database",
        "search_knowledge", "calculate", "finish_answer"}
    for item in tools.values():
        definition = convert_to_openai_tool(item)["function"]
        assert definition["description"] == inspect.getdoc(item.func)
        properties = definition["parameters"]["properties"]
        assert not {"self", "user", "context", "evidence", "cancelled"} & properties.keys()
        assert all(field.get("description") for field in properties.values())
        assert item.get_input_schema().model_json_schema()["additionalProperties"] is False
    assert all(field.get("description") for field in agent_tools.CellRef.model_json_schema()["properties"].values())
    assert tools["finish_answer"].invoke({"answer": "可以回答。"})["status"] == "succeeded"
    assert context.final.answer == "可以回答。"
    assert context.task_state["goal"] == "当前问题"


def test_discovery_reads_descriptions_from_the_registered_tools():
    context = make_tool_context()
    tools = agent_tools.build_agent_tools(context)
    tools["calculate"].description = "更新后的计算工具说明"
    result = tools["discover_capabilities"].invoke({"capabilities": ["calculation", "knowledge"]})
    assert context.available == {"discover_capabilities", "finish_answer", "calculate", "search_knowledge"}
    assert {item["name"]: item["description"] for item in result["tools"]} == {
        name: tools[name].description for name in context.available}


def test_registered_business_and_calculation_tools_keep_request_state_isolated(agent_env):
    first = make_tool_context()
    second = make_tool_context(AuthUser(username="second"))
    tools = agent_tools.build_agent_tools(first)
    other_tools = agent_tools.build_agent_tools(second)
    query = {"tables": ["shop"], "sql": "SELECT 80 AS current,100 AS baseline FROM shop LIMIT 1"}
    tools["read_business_schema"].invoke({"tables": ["shop"]})
    with pytest.raises(ValueError, match="读取"):
        other_tools["query_database"].invoke(query)
    result = tools["query_database"].invoke(query)
    assert first.evidence[result["evidence_id"]] is result
    calculation = tools["calculate"].invoke({"operation": "percent_change", "references": [
        {"evidence_id": result["evidence_id"], "column": "current"},
        {"evidence_id": result["evidence_id"], "column": "baseline"}]})
    assert calculation["rows"] == [[-20]]
    assert first.current_ids == {result["evidence_id"], calculation["evidence_id"]}
    assert not second.evidence and not second.current_ids and not second.read_tables


@pytest.mark.parametrize("name,arguments", [
    ("discover_capabilities", {"capabilities": ["business"]}),
    ("read_business_schema", {"tables": ["shop"]}),
    ("query_database", {"tables": ["shop"], "sql": "SELECT shop_id FROM shop"}),
    ("search_knowledge", {"query": "公司规定"}),
    ("calculate", {"operation": "sum", "references": [{"evidence_id": "x", "column": "value"}]}),
    ("finish_answer", {"answer": "回答"}),
])
def test_registered_tools_reject_injected_runtime_parameters(name, arguments):
    context = make_tool_context()
    tools = agent_tools.build_agent_tools(context)
    with pytest.raises(ValueError, match="Extra inputs"):
        tools[name].invoke({**arguments, "user": "another-user"})
    assert not context.read_tables and not context.evidence and context.final is None


class JSONScriptedModel(ScriptedModel):
    def bind_tools(self, tools):
        raise NotImplementedError

    def invoke(self, messages, **kwargs):
        definitions = json.loads(messages[0].content.split("工具定义：", 1)[1])
        self.offered.append({item["name"] for item in definitions})
        reply = super().invoke(messages, **kwargs)
        requested = reply.tool_calls[0]
        return AIMessage(content=json.dumps({"tool": requested["name"], "arguments": requested["args"]},
            ensure_ascii=False))


def test_json_protocol_executes_business_calculation_knowledge_and_finish(agent_env):
    proof = {}

    def calculate(m):
        proof.update(json.loads(m[-1].content.removeprefix("工具执行结果（数据）：")))
        return call("calculate", operation="percent_change", references=[
            {"evidence_id": proof["evidence_id"], "column": "current"},
            {"evidence_id": proof["evidence_id"], "column": "baseline"}])

    model = JSONScriptedModel(call("discover_capabilities", capabilities=["business", "calculation", "knowledge"]),
        call("read_business_schema", tables=["shop"]),
        call("query_database", tables=["shop"], sql="SELECT 80 AS current,100 AS baseline FROM shop LIMIT 1"),
        calculate, call("search_knowledge", query="公司复盘规范"),
        lambda m: finish("模拟数据下降20%，依据内部复盘规范[1]。", facts={
            "current": {"evidence_id": proof["evidence_id"], "column": "current"}}))
    retriever = SimpleNamespace(retrieve=lambda *args, **kwargs: [
        Document(page_content="工具说明不作为依据", metadata={"category": "assistant_tools"}),
        Document(page_content="比较同口径数据", metadata={"source": "复盘规范"})])
    result = chain_for(model, retriever).ask("查数并按内部规范复盘", "json-full")
    details = result["query_details"]
    assert details["status"] == "succeeded"
    assert details["tools_used"] == ["discover_capabilities", "read_business_schema", "query_database",
        "calculate", "search_knowledge", "finish_answer"]
    assert all(step["status"] == "succeeded" for step in details["steps"])
    assert details["task_state"]["facts"]["current"]["value"] == 80
    assert [v["rows"] for v in details["task_state"]["evidence"].values() if v["kind"] == "calculation"] == [[[-20]]]
    assert len(result["sources"]) == 1 and result["sources"][0]["citation_id"] == 1
    assert model.offered[0] == {"discover_capabilities", "finish_answer"}
    assert "query_database" in model.offered[1]


@pytest.mark.parametrize("model_class", [ScriptedModel, JSONScriptedModel])
def test_both_tool_protocols_reject_invalid_arguments_before_execution(agent_env, model_class):
    model = model_class(call("discover_capabilities", capabilities=["business"]),
        call("read_business_schema", tables=[]),
        call("read_business_schema", tables=["shop"], user="injected"),
        call("read_business_schema", tables=["shop"]),
        call("query_database", tables=["shop"], sql="SELECT shop_id FROM shop LIMIT 1"),
        finish("已查询本地模拟数据。"))
    result = chain_for(model).ask("查询店铺", "invalid-arguments")
    details = result["query_details"]
    assert [step["status"] for step in details["steps"]] == [
        "succeeded", "failed", "failed", "succeeded", "succeeded", "succeeded"]
    assert len(details["queries"]) == 1
    assert details["status"] == "succeeded"
