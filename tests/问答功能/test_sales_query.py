"""销量 Text-to-SQL 的数值、只读边界、对话/API和页面验证，不调用真实模型。"""
from datetime import date, datetime, timezone
import json
import re
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import sessionmaker
from streamlit.testing.v1 import AppTest

from api.dependencies import get_rag_chain_dep
from api.routes.qa import router
from config.settings import settings
from core import database
from core.auth import ANONYMOUS_USER
from core.db_models import LiveSales, QAInteraction, VideoSales
from core.intent_recognizer import IntentRecognizer
from core.memory_manager import MemoryManager
from core.rag_chain import RAGChain
from core.sales_query import SalesQueryError, SalesQueryService, is_sales_question, validate_sql
from core.sales_seed import seed_sales
from core.sales_schema_retriever import (SalesSchemaError, SalesSchemaRetriever,
    build_sales_schema_index, load_catalog)
from frontend.qa_stream import read_answer_events


COMBINED = """WITH channels AS (
    SELECT anchor_id, sales_quantity AS live_quantity, 0 AS video_quantity FROM live_sales
    UNION ALL SELECT anchor_id, 0, sales_quantity FROM video_sales
) SELECT anchor_id AS 主播ID, SUM(live_quantity) AS 直播销量,
    SUM(video_quantity) AS 视频销量, SUM(live_quantity + video_quantity) AS 合计销量
    FROM channels GROUP BY anchor_id ORDER BY 合计销量 DESC, 主播ID LIMIT 5"""


@pytest.fixture
def sales_db(tmp_path, monkeypatch, schema_index):
    monkeypatch.setattr(settings, "AGENT_ENABLED", False)
    engine = create_engine(f"sqlite:///{tmp_path / 'sales.db'}", connect_args={"check_same_thread": False})
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "SessionLocal", sessionmaker(engine, expire_on_commit=False))
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", False)
    monkeypatch.setattr(settings, "AUTH_REQUIRED", False)
    monkeypatch.setattr(settings, "SALES_QUERY_ENABLED", True)
    monkeypatch.setattr(settings, "INTENT_ROUTER_ENABLED", False)
    monkeypatch.setattr(settings, "QUALITY_GUARD_ENABLED", False)
    monkeypatch.setattr(settings, "REQUIRE_EVIDENCE", True)
    monkeypatch.setattr(settings, "ALLOW_GENERAL_ANSWERS", False)
    monkeypatch.setattr("core.sales_query.get_sales_schema_retriever", lambda: schema_index)
    database.init_db()
    common = {"period_start": date(2026, 9, 16), "period_end": date(2026, 10, 15),
        "updated_at": datetime(2026, 10, 16, tzinfo=timezone.utc)}
    with engine.begin() as db:
        db.execute(LiveSales.__table__.insert(), [dict(common, room_id=room, anchor_id=anchor, sales_quantity=qty)
            for room, anchor, qty in [("R1", "A001", 10), ("R2", "A001", 20), ("R3", "A002", 0), ("R4", "A003", 15)]])
        db.execute(VideoSales.__table__.insert(), [dict(common, video_id=video, anchor_id=anchor, sales_quantity=qty)
            for video, anchor, qty in [("V1", "A001", 5), ("V2", "A001", 7), ("V3", "A004", 100), ("V4", "A002", 0)]])
    yield engine
    engine.dispose()


class SchemaTestEmbeddings:
    """测试使用确定向量，实际经过独立 Chroma 检索，不下载模型。"""
    def embed_query(self, text):
        live = bool(re.search(r"直播|房间|live_sales|开播", text))
        video = bool(re.search(r"视频|作品|video_sales", text))
        sales = bool(re.search(r"销量|主播|带货|成交件数|卖", text))
        return [float(sales), float(live) * 2, float(video) * 2,
            float("30天" in text) * 0.5, float("UNION ALL" in text) * 0.5,
            float(not (live or video or sales))]

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


@pytest.fixture(scope="module")
def schema_index(tmp_path_factory):
    directory = tmp_path_factory.mktemp("sales_schema")
    model = SchemaTestEmbeddings()
    build_sales_schema_index(index_directory=directory, embedding_model=model)
    return SalesSchemaRetriever(index_directory=directory, embedding_model=model)


def model_for(*plans):
    return FakeListChatModel(responses=[json.dumps(plan, ensure_ascii=False) for plan in plans])


def final_result(service, question="近30天直播加视频销量最高的五个主播是谁？", history=None):
    return list(service.events(question, history or []))[-1]


def make_chain(model):
    chain = object.__new__(RAGChain)
    chain.llm = model
    chain.intent_recognizer = IntentRecognizer()
    chain.memory_manager = MemoryManager()
    def forbid_retrieval(*args, **kwargs):
        raise AssertionError("销量查询不应检索文档")
    chain.retriever = SimpleNamespace(get_compresstion_retriever=forbid_retrieval)
    chain._rag_chain = RunnableLambda(forbid_retrieval)
    return chain


def make_app(chain):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_rag_chain_dep] = lambda: chain
    return app


def test_combined_totals_do_not_multiply_and_keep_single_channel_anchors(sales_db):
    result = final_result(SalesQueryService(model_for({"sql": COMBINED})))
    assert result["query_details"]["status"] == "succeeded"
    assert result["query_details"]["rows"] == [["A004", 0, 100, 100], ["A001", 30, 12, 42],
        ["A003", 15, 0, 15], ["A002", 0, 0, 0]]
    assert result["evidence_sufficient"] is True
    assert result["citation_status"] == "database"
    assert result["query_details"]["schema_retrieval"]["tables"] == ["live_sales", "video_sales"]
    assert result["query_details"]["schema_retrieval"]["status"] == "validated"


@pytest.mark.parametrize("sql", [
    "DELETE FROM live_sales", "UPDATE video_sales SET sales_quantity = 0",
    "SELECT * FROM live_sales; DROP TABLE video_sales",
    "SELECT * FROM users", "SELECT * FROM sqlite_master", "SELECT * FROM main.live_sales",
    "SELECT * FROM live_sales WHERE EXISTS (SELECT 1 FROM users)",
    "WITH secrets AS (SELECT * FROM users) SELECT * FROM secrets",
    "WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM x) SELECT * FROM x, live_sales",
    "SELECT load_extension('anything') FROM live_sales",
    "SELECT randomblob(999999999) FROM live_sales", "PRAGMA query_only=OFF",
    "ATTACH DATABASE 'other.db' AS other", "SELECT * FROM live_sales INTO other",
    "SELECT * FROM live_sales l JOIN video_sales v ON l.anchor_id=v.anchor_id",
    "SELECT * FROM live_sales LIMIT -1", "SELECT * FROM live_sales LIMIT 1 OFFSET 999999",
])
def test_sql_guard_rejects_unsafe_or_out_of_scope_queries(sql):
    with pytest.raises(SalesQueryError):
        validate_sql(sql)


@pytest.mark.parametrize("sql", ["SELECT * FROM users", "DELETE FROM live_sales", "PRAGMA query_only=OFF",
    "ATTACH DATABASE ':memory:' AS other", "SELECT load_extension('anything') FROM live_sales"])
def test_database_authorizer_blocks_operations_even_without_sql_guard(sales_db, sql):
    service = SalesQueryService(None)
    with pytest.raises(SalesQueryError):
        with service.readonly_connection() as connection:
            connection.execute(sql).fetchall()
    with sales_db.connect() as connection:
        assert connection.scalar(select(func.sum(LiveSales.sales_quantity))) == 45


def test_file_connection_stays_readonly_even_if_upper_guards_are_removed(sales_db):
    with pytest.raises(SalesQueryError):
        with SalesQueryService(None).readonly_connection() as connection:
            connection.set_authorizer(None)
            connection.execute("PRAGMA query_only=OFF")
            connection.execute("UPDATE live_sales SET sales_quantity=999")
    with sales_db.connect() as connection:
        assert connection.scalar(select(func.sum(LiveSales.sales_quantity))) == 45


def test_valid_sql_has_unknown_columns_rejected():
    with pytest.raises(SalesQueryError) as error:
        validate_sql("SELECT secret FROM live_sales")
    assert error.value.repairable


@pytest.mark.parametrize("question", ["昨天直播销量多少？", "近7天视频销量多少？", "上周销量排名", "视频销售额多少？",
    "直播退款率多少？", "每天的销量趋势？"])
def test_unsupported_periods_and_metrics_never_call_model(sales_db, question):
    result = final_result(SalesQueryService(None), question)
    assert result["response_type"] == "clarify"
    assert not result["query_details"]["attempts"]


def test_absolute_dates_must_match_the_entire_snapshot(sales_db):
    result = final_result(SalesQueryService(None), "2026-10-15的直播销量多少？")
    assert result["query_details"]["code"] == "unsupported_request"
    valid = final_result(SalesQueryService(model_for({"sql": "SELECT SUM(sales_quantity) FROM live_sales"})),
        "2026-09-16至2026-10-15的直播销量多少？")
    assert valid["query_details"]["rows"] == [[45]]


def test_zero_and_no_matching_records_are_distinct(sales_db):
    service = SalesQueryService(model_for({"sql": "SELECT sales_quantity FROM live_sales WHERE room_id='R3'"},
        {"sql": "SELECT sales_quantity FROM live_sales WHERE room_id='missing'"}))
    zero = final_result(service)
    empty = final_result(service)
    assert zero["query_details"]["rows"] == [[0]]
    assert "未找到" not in zero["answer"]
    assert empty["query_details"]["rows"] == []
    assert "不代表销量为零" in empty["answer"]


def test_count_without_column_is_allowed_only_for_sales_tables(sales_db):
    result = final_result(SalesQueryService(model_for({"sql": "SELECT COUNT(*) AS 房间数 FROM live_sales"})))
    assert result["query_details"]["rows"] == [[4]]
    with pytest.raises(SalesQueryError):
        with SalesQueryService(None).readonly_connection() as connection:
            connection.execute("SELECT COUNT(*) FROM users").fetchall()


def test_case_aggregation_and_numeric_thresholds(sales_db):
    sql = "SELECT SUM(CASE WHEN sales_quantity > 0 THEN 1 ELSE 0 END) AS 有销量房间数 FROM live_sales"
    result = final_result(SalesQueryService(model_for({"sql": sql})))
    assert result["query_details"]["rows"] == [[3]]


def test_boolean_filters_and_exists_are_sql_constructs_not_callable_functions(sales_db):
    sql = """SELECT SUM(sales_quantity) AS total FROM live_sales
        WHERE (anchor_id='A001' OR anchor_id='A003') AND period_start='2026-09-16'
        AND EXISTS (SELECT 1 FROM video_sales WHERE anchor_id='A001')"""
    result = final_result(SalesQueryService(model_for({"sql": sql})))
    assert result["query_details"]["rows"] == [[45]]


def test_only_one_repair_and_both_attempts_are_audited(sales_db):
    result = final_result(SalesQueryService(model_for({"sql": "SELECT unknown FROM live_sales"},
        {"sql": "SELECT SUM(sales_quantity) AS total FROM live_sales"})))
    assert result["query_details"]["rows"] == [[45]]
    assert [item["status"] for item in result["query_details"]["attempts"]] == ["rejected", "succeeded"]
    failed = final_result(SalesQueryService(model_for({"sql": "SELECT unknown FROM live_sales"})))
    assert len(failed["query_details"]["attempts"]) == 2
    assert failed["query_details"]["status"] == "failed"


def test_unsafe_sql_is_not_repaired_or_executed(sales_db):
    result = final_result(SalesQueryService(model_for({"sql": "SELECT * FROM users"})))
    assert len(result["query_details"]["attempts"]) == 1
    assert result["query_details"]["status"] == "refused"
    assert result["query_details"]["sql"] is None


def test_row_limit_is_enforced_and_disclosed(sales_db, monkeypatch):
    monkeypatch.setattr(settings, "SALES_QUERY_MAX_ROWS", 2)
    result = final_result(SalesQueryService(model_for({"sql": "SELECT * FROM live_sales ORDER BY room_id LIMIT 99999"})))
    assert result["query_details"]["row_count"] == 2
    assert result["query_details"]["truncated"]
    assert "返回数量或大小上限" in result["answer"]


def test_timeout_interrupts_expensive_query(sales_db, monkeypatch):
    import core.sales_query as module
    service = SalesQueryService(None)
    clock = iter([0.0] + [10.0] * 100)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    with pytest.raises(SalesQueryError) as error:
        with service.readonly_connection() as connection:
            connection.execute("SELECT SUM(a.sales_quantity) FROM live_sales a, live_sales b, live_sales c, live_sales d, live_sales e").fetchall()
    assert error.value.code == "timeout"


def test_inconsistent_snapshots_stop_before_generation(sales_db):
    with sales_db.begin() as connection:
        connection.execute(text("UPDATE video_sales SET period_end='2026-10-14'"))
    result = final_result(SalesQueryService(None))
    assert result["query_details"]["code"] == "inconsistent_snapshot"


def test_routing_respects_last_turn_and_document_questions():
    sales_history = [HumanMessage(content="查直播销量"), AIMessage(content="结果", additional_kwargs={"query_details": {"status": "succeeded"}})]
    assert is_sales_question("那前十呢？", sales_history)
    assert is_sales_question("他的视频销量呢？", sales_history)
    assert not is_sales_question("报销怎么办？", sales_history)
    assert not is_sales_question("如何提高直播销量？", sales_history)
    assert not is_sales_question("那前十呢？", sales_history + [HumanMessage(content="报销流程"), AIMessage(content="文档答案")])


@pytest.mark.parametrize("streaming", [False, True])
def test_api_persists_sql_and_results_and_restores_followup_context(sales_db, streaming):
    model = model_for({"sql": COMBINED}, {"sql": "SELECT SUM(sales_quantity) AS 视频销量 FROM video_sales WHERE anchor_id='A004'"})
    chain = make_chain(model)
    client = TestClient(make_app(chain))
    if streaming:
        response = client.post("/qa/ask/stream", json={"question": "直播和视频总销量前五主播", "session_id": "sales"})
        events = list(read_answer_events(SimpleNamespace(iter_lines=lambda **kwargs: response.iter_lines())))
        data = events[-1]["data"]
        assert events[-1]["event"] == "done"
        assert any(item["event"] == "status" and "只读" in item["data"]["message"] for item in events)
    else:
        data = client.post("/qa/ask", json={"question": "直播和视频总销量前五主播", "session_id": "sales"}).json()
    assert data["query_details"]["rows"][0] == ["A004", 0, 100, 100]
    assert data["response_type"] == "data_query"
    history = client.get("/qa/history", params={"session_id": "sales"}).json()["history"]
    assert history[-1]["query_details"] == data["query_details"]
    result = chain.ask("他的视频销量呢？", "sales", user=ANONYMOUS_USER)
    assert result["query_details"]["rows"] == [[100]]
    with database.session_scope() as db:
        records = list(db.scalars(select(QAInteraction).order_by(QAInteraction.created_at)))
        assert len(records) == 2
        assert all(row.query_details["status"] == "succeeded" for row in records)


def test_clarification_not_replaced_by_document_evidence_refusal(sales_db):
    chain = make_chain(model_for({"clarification": "请提供主播ID。"}))
    result = chain.ask("查一下主播销量", persist=False)
    assert result["answer"] == "请提供主播ID。"
    result = chain.ask("昨天直播销量多少", persist=False)
    assert "汇总快照" in result["answer"]


def test_seed_is_repeatable_and_never_overwrites_existing_data(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'seed.db'}")
    try:
        summary = seed_sales(engine, date(2026, 10, 15))
        assert (summary["live_rows"], summary["video_rows"], summary["anchors"]) == (40, 120, 20)
        assert (summary["period_start"], summary["period_end"]) == ("2026-09-16", "2026-10-15")
        assert summary["live_total"] == 176925
        assert summary["video_total"] == 179088
        with pytest.raises(ValueError, match="已有数据"):
            seed_sales(engine)
        with engine.connect() as connection:
            assert connection.scalar(select(func.count()).select_from(LiveSales)) == 40
            assert connection.scalar(select(func.count()).select_from(VideoSales)) == 120
            live = set(connection.scalars(select(LiveSales.anchor_id)))
            video = set(connection.scalars(select(VideoSales.anchor_id)))
            assert live - video == {"A019"}
            assert video - live == {"A020"}
    finally:
        engine.dispose()


def test_sales_table_and_sql_render_in_page(sales_db):
    details = final_result(SalesQueryService(model_for({"sql": COMBINED})))["query_details"]
    app = AppTest.from_string("from frontend.sales_query import render_sales_query\nimport streamlit as st\nrender_sales_query(st.session_state.details)")
    app.session_state["details"] = details
    app.run(timeout=20)
    assert not app.exception
    assert len(app.dataframe) == 1
    assert app.dataframe[0].value.iloc[0].tolist() == ["A004", 0, 100, 100]
    assert app.code[0].value == details["sql"]


def test_existing_qa_table_receives_nullable_metadata_column(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE qa_interactions(id VARCHAR(32) PRIMARY KEY)"))
        from unittest.mock import patch
        with patch.object(database, "engine", engine):
            database._ensure_schema_compatibility()
        with engine.connect() as connection:
            columns = {row[1] for row in connection.execute(text("PRAGMA table_info(qa_interactions)"))}
            assert "query_details" in columns
    finally:
        engine.dispose()


def test_video_only_prompt_comes_from_retrieved_schema(sales_db):
    prompts = []
    def plan(messages):
        prompts.append(messages[0].content)
        return AIMessage(content=json.dumps({"sql": "SELECT SUM(sales_quantity) AS 视频销量 FROM video_sales"}))
    result = final_result(SalesQueryService(RunnableLambda(plan)), "视频总销量多少？")
    assert result["query_details"]["rows"] == [[112]]
    retrieval = result["query_details"]["schema_retrieval"]
    assert retrieval["tables"] == ["video_sales"]
    assert set(retrieval["rules"]) == {"rule.snapshot", "rule.aggregation"}
    assert "video_sales(" in prompts[0]
    assert "live_sales" not in prompts[0]
    assert "video_id：视频ID" in prompts[0]
    assert "记录粒度" in prompts[0] and "不能用 DISTINCT" in prompts[0]


@pytest.mark.parametrize("question, expected", [
    ("直播房间销量排行", ["live_sales"]),
    ("短视频销量排行", ["video_sales"]),
    ("作品带货件数最多的是谁", ["video_sales"]),
    ("主播直播与视频销量比较", ["live_sales", "video_sales"]),
    ("主播总销量排行", ["live_sales", "video_sales"]),
])
def test_metadata_retrieval_covers_requested_channels(schema_index, question, expected):
    selection = schema_index.retrieve(question, [])
    assert selection.table_names == expected
    assert len(selection.trace["rules"]) == 2
    assert all(hit.get("reason") for hit in selection.trace["hits"])


def test_followup_inherits_scope_and_current_channel_overrides_it(schema_index):
    history = [HumanMessage(content="主播总销量排行"), AIMessage(content="销量结果",
        additional_kwargs={"query_details": {"schema_retrieval": {"tables": ["live_sales", "video_sales"]}}})]
    assert schema_index.retrieve("那前十呢？", history).table_names == ["live_sales", "video_sales"]
    assert schema_index.retrieve("那他的视频呢？", history).table_names == ["video_sales"]
    single = [AIMessage(content="视频结果", additional_kwargs={"query_details": {
        "schema_retrieval": {"tables": ["video_sales"]}}})]
    assert schema_index.retrieve("按主播降序", single).table_names == ["video_sales"]
    assert schema_index.retrieve("那两个渠道一起合计呢？", single).table_names == ["live_sales", "video_sales"]
    assert schema_index.retrieve("那加上直播呢？", single).table_names == ["live_sales", "video_sales"]
    # 最近一轮已经切换到文档话题，就不能沿用更早的查数范围。
    with pytest.raises(SalesSchemaError, match="未检索到"):
        schema_index.retrieve("那前十呢？", single + [AIMessage(content="普通文档答复")])


def test_followup_can_restore_scope_from_older_saved_sql(schema_index):
    history = [AIMessage(content="旧版查询结果", additional_kwargs={"query_details": {
        "sql": "SELECT anchor_id, SUM(sales_quantity) FROM video_sales GROUP BY anchor_id"}})]
    assert schema_index.retrieve("那前十呢？", history).table_names == ["video_sales"]
    assert schema_index.retrieve("改查直播销量", history).table_names == ["live_sales"]


def test_sql_cannot_access_an_unretrieved_sales_table(sales_db):
    result = final_result(SalesQueryService(model_for({"sql": "SELECT SUM(sales_quantity) FROM live_sales"})),
        "视频销量多少？")
    assert result["query_details"]["code"] == "unretrieved_table"
    assert result["query_details"]["sql"] is None
    assert len(result["query_details"]["attempts"]) == 1
    with pytest.raises(SalesQueryError):
        validate_sql(COMBINED, allowed_tables=["video_sales"])


def test_execution_authorizer_also_limits_selected_tables(sales_db, schema_index):
    service = SalesQueryService(None)
    with service.readonly_connection() as connection:
        snapshot = service.snapshot(connection)
    selection = schema_index.retrieve("视频销量", [])
    with pytest.raises(SalesQueryError):
        service.execute("SELECT SUM(sales_quantity) FROM live_sales", snapshot, selection=selection)


def test_index_missing_never_calls_model_or_falls_back_to_all_tables(sales_db, tmp_path):
    service = SalesQueryService(None, schema_retriever=SalesSchemaRetriever(index_directory=tmp_path / "missing"))
    result = final_result(service, "视频销量多少？")
    assert result["query_details"]["code"] == "schema_index_unavailable"
    assert result["query_details"]["attempts"] == []
    assert result["query_details"]["schema_retrieval"]["status"] == "failed"
    assert not (tmp_path / "missing").exists()


def test_irrelevant_question_has_no_schema_match(schema_index):
    with pytest.raises(SalesSchemaError) as error:
        schema_index.retrieve("明天的天气如何", [])
    assert error.value.code == "schema_not_found"


def test_database_schema_drift_stops_before_generation(sales_db):
    with sales_db.begin() as connection:
        connection.execute(text("ALTER TABLE video_sales RENAME COLUMN sales_quantity TO old_quantity"))
    result = final_result(SalesQueryService(None), "视频销量多少？")
    assert result["query_details"]["code"] == "schema_changed"
    assert result["query_details"]["attempts"] == []
    assert result["query_details"]["schema_retrieval"]["status"] == "failed"


def test_schema_is_rechecked_after_model_generation(sales_db):
    def change_schema(messages):
        with sales_db.begin() as connection:
            connection.execute(text("ALTER TABLE video_sales ADD COLUMN extra TEXT"))
        return AIMessage(content=json.dumps({"sql": "SELECT SUM(sales_quantity) FROM video_sales"}))
    result = final_result(SalesQueryService(RunnableLambda(change_schema)), "视频总销量")
    assert result["query_details"]["code"] == "schema_changed"
    assert "rows" not in result["query_details"]


def test_catalog_update_requires_index_rebuild_and_rebuild_is_repeatable(tmp_path):
    catalog, _, _ = load_catalog()
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
    directory = tmp_path / "index"
    model = SchemaTestEmbeddings()
    first = build_sales_schema_index(index_directory=directory, catalog_path=path, embedding_model=model)
    retriever = SalesSchemaRetriever(index_directory=directory, catalog_path=path, embedding_model=model)
    assert retriever.retrieve("视频销量", []).table_names == ["video_sales"]
    catalog["tables"][1]["description"] += " 新的说明。"
    path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SalesSchemaError) as error:
        retriever.retrieve("视频销量", [])
    assert error.value.code == "schema_index_stale"
    second = build_sales_schema_index(index_directory=directory, catalog_path=path, embedding_model=model)
    third = build_sales_schema_index(index_directory=directory, catalog_path=path, embedding_model=model)
    assert first["catalog_hash"] != second["catalog_hash"] == third["catalog_hash"]
    assert retriever.retrieve("视频销量", []).table_names == ["video_sales"]
    assert third["documents"] == 4


def test_missing_required_rule_blocks_generation(sales_db, tmp_path):
    from core.sales_schema_retriever import _client, COLLECTION_NAME

    directory = tmp_path / "incomplete"
    model = SchemaTestEmbeddings()
    build_sales_schema_index(index_directory=directory, embedding_model=model)
    _client(str(directory.resolve())).get_collection(COLLECTION_NAME).delete(ids=["rule.aggregation"])
    service = SalesQueryService(None,
        schema_retriever=SalesSchemaRetriever(index_directory=directory, embedding_model=model))
    result = final_result(service, "视频销量")
    assert result["query_details"]["code"] == "schema_index_incomplete"
    assert not result["query_details"]["attempts"]


def test_missing_channel_in_retrieval_does_not_fall_back_to_catalog(sales_db, schema_index, monkeypatch):
    from core.sales_schema_retriever import _client, COLLECTION_NAME

    collection = _client(str(schema_index.index_directory.resolve())).get_collection(COLLECTION_NAME)
    original_query = type(collection).query
    def omit_live(self, *args, **kwargs):
        result = original_query(self, *args, **kwargs)
        keep = [i for i, key in enumerate(result["ids"][0]) if key != "table.live_sales"]
        for key in ("ids", "documents", "metadatas", "distances"):
            result[key] = [[result[key][0][i] for i in keep]]
        return result
    monkeypatch.setattr(type(collection), "query", omit_live)
    result = final_result(SalesQueryService(None), "直播和视频的总销量")
    assert result["query_details"]["code"] == "schema_not_found"
    assert not result["query_details"]["attempts"]


def test_rebuild_supports_changed_embedding_dimension(tmp_path):
    class WiderEmbeddings(SchemaTestEmbeddings):
        def embed_query(self, text):
            return super().embed_query(text) + [0.0]
    directory = tmp_path / "dimension"
    build_sales_schema_index(index_directory=directory, embedding_model=SchemaTestEmbeddings())
    wider = WiderEmbeddings()
    build_sales_schema_index(index_directory=directory, embedding_model=wider)
    assert SalesSchemaRetriever(index_directory=directory, embedding_model=wider).retrieve("视频销量", []).table_names == ["video_sales"]
