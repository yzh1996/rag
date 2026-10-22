"""Dictionary integrity and business behavior; unit fixtures never touch local business data."""
from datetime import date
import json

import pytest
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.pool import StaticPool

from core.business_schema import business_metadata, load_business_catalog, query_schema, schema_status
from core.business_seed import build_records, seed_business_data
from core.business_query import validate_business_sql


@pytest.fixture
def business_engine():
    # A process-local fixture only; persistent synthetic data is seeded in MySQL.
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    seed_business_data(engine, date(2026, 10, 20), days=8)
    yield engine
    engine.dispose()


def test_dictionary_matches_source_without_placeholder_columns():
    from scripts.extract_business_dictionary import extract_dictionary, ROOT
    source = ROOT / "doc" / "六个维度大数据平台数据表与字段字典.docx"
    assert extract_dictionary(source) == load_business_catalog()
    status = schema_status()
    assert status["dictionary_tables"] == 64
    assert status["creatable_tables"] == 49
    assert len(status["pending_tables"]) == 15
    metadata = business_metadata()
    assert "live_sales" not in metadata.tables and "demo_live_daily" not in metadata.tables
    assert all("源文档未提取到字段" not in table.c for table in metadata.tables.values())
    assert "price" not in query_schema({"ads_product_live_rank"})["ads_product_live_rank"]


def test_create_database_connects_without_selecting_missing_database(monkeypatch, capsys):
    from scripts import seed_mysql_business as script
    from sqlalchemy.engine import make_url

    connections, statements = [], []

    class Engine:
        def connect(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def exec_driver_sql(self, statement):
            statements.append(statement)

        def dispose(self):
            pass

    def create_test_engine(url, **options):
        connections.append(url)
        return Engine()

    monkeypatch.setattr(script.settings, "DATABASE_URL",
                        "mysql+pymysql://test_user:test_password@127.0.0.1:3306/new_business")
    monkeypatch.setattr(script, "create_engine", create_test_engine)
    monkeypatch.setattr(script, "seed_business_data", lambda *args: {"created": True})
    monkeypatch.setattr("sys.argv", ["seed_mysql_business.py", "--create-database"])
    script.main()

    assert not make_url(connections[0]).database
    assert make_url(connections[1]).database == "new_business"
    assert statements == ["CREATE DATABASE IF NOT EXISTS `new_business` CHARACTER SET utf8mb4 COLLATE utf8mb4_bin"]
    assert json.loads(capsys.readouterr().out)["created"] is True


def test_shop_latest_phrase_uses_commerce_intent():
    from core.commerce_assistant import is_commerce_analysis_question
    assert is_commerce_analysis_question("帮我完整复盘清研美妆旗舰店最新一场直播数据，看看整体卖得怎么样")


def test_seed_is_repeatable_and_refuses_modified_or_existing_records(business_engine):
    result = seed_business_data(business_engine, date(2026, 10, 20), days=8)
    assert result["created"] is False
    assert result["counts"]["shop"] == 6
    with business_engine.begin() as c:
        c.execute(text("INSERT INTO shop(shop_id,create_time,name) VALUES (999,'2026-10-20','已有店铺')"))
    with pytest.raises(ValueError, match="数量发生变化"):
        seed_business_data(business_engine)
    engine = create_engine("sqlite://")
    metadata = business_metadata()
    metadata.tables["shop"].create(engine)
    with engine.begin() as c:
        c.execute(text("INSERT INTO shop(shop_id,create_time,name) VALUES (1,'2026-10-20','真实店铺')"))
    with pytest.raises(ValueError, match="已有数据"):
        seed_business_data(engine)
    engine.dispose()


def test_live_and_summary_totals_come_from_same_records(business_engine):
    with business_engine.connect() as c:
        expected = c.scalar(text("SELECT SUM(number*min_price) FROM live_product_sale"))
        assert expected == c.scalar(text("SELECT SUM(sales) FROM live_sales_info"))
        assert expected == c.scalar(text("SELECT SUM(sales) FROM dwd_live_sales_info"))
        assert c.scalar(text("SELECT COUNT(*) FROM ads_live_product_sale_info")) == 0
        combined = expected + c.scalar(text("SELECT SUM(sales) FROM date_product_video_incr"))
        assert combined == c.scalar(text("SELECT SUM(sales) FROM ads_shop_sale_rank"))
        orphans = c.scalar(text("SELECT COUNT(*) FROM live_base l LEFT JOIN jl_user u ON u.anchor_id=l.anchor_id WHERE u.anchor_id IS NULL"))
        assert orphans == 0


@pytest.mark.parametrize("sql", [
    "DELETE FROM shop", "SELECT * FROM shop; DROP TABLE shop", "SELECT * FROM mysql.user",
    "SELECT SLEEP(1) FROM shop", "SELECT LOAD_FILE('/etc/passwd') FROM shop",
    "SELECT name FROM shop INTO OUTFILE '/tmp/leak'", "SELECT @@version FROM shop",
    "SELECT no_such_field FROM shop", "SELECT * FROM business_datasets", "SELECT * FROM demo_anchors",
    "SELECT price FROM ads_product_live_rank", "SELECT * FROM shop FOR UPDATE",
    "WITH RECURSIVE x AS (SELECT shop_id FROM shop UNION ALL SELECT shop_id FROM x) SELECT * FROM x",
])
def test_sql_denies_writes_system_access_and_undocumented_fields(sql):
    with pytest.raises(ValueError):
        validate_business_sql(sql, query_schema())


def test_sql_allows_documented_join_and_bounds_result():
    sql, tables = validate_business_sql("SELECT s.name,SUM(i.sales) AS amount FROM shop s "
        "JOIN jl_user u ON u.shop_id=s.shop_id JOIN live_sales_info i ON i.anchor_id=u.anchor_id "
        "GROUP BY s.shop_id,s.name ORDER BY amount DESC", query_schema(), max_rows=10)
    assert sql.endswith("LIMIT 11")
    assert tables == ["jl_user", "live_sales_info", "shop"]


def test_mysql_boolean_storage_mapping_keeps_other_type_checks():
    from sqlalchemy import Boolean, Integer, String
    from sqlalchemy.dialects.mysql import TINYINT
    from core.business_query import _storage_types_match

    assert _storage_types_match(TINYINT(display_width=1), Boolean())
    assert not _storage_types_match(TINYINT(display_width=4), Boolean())
    assert not _storage_types_match(Integer(), Boolean())
    assert not _storage_types_match(String(1), Boolean())
    assert _storage_types_match(Integer(), Integer())


@pytest.mark.parametrize("question", [
    "完整复盘清研美妆旗舰店最新一场直播",
    "复盘不存在店铺最新一场直播",
    "分析日光美妆店近30天直播表现",
])
def test_shop_questions_use_general_analysis_and_ask_for_anchor(monkeypatch, question):
    from types import SimpleNamespace
    from core import commerce_assistant as module
    from core.auth import ANONYMOUS_USER
    from langchain_core.messages import AIMessage
    monkeypatch.setattr("core.business_reader.database.engine",
                        SimpleNamespace(dialect=SimpleNamespace(name="mysql")))
    monkeypatch.setattr(module, "discover_tool_ids", lambda *args, **kwargs: ["resolve_anchor", "analyze_anchor_live"])
    monkeypatch.setattr(module, "find_anchor_mentions", lambda value: [])
    monkeypatch.setattr(module, "find_anchors", lambda value: [])
    model_questions, offered_tools = [], []

    class LLM:
        def bind_tools(self, tools):
            offered_tools.extend(tool.name for tool in tools)
            return self

        def invoke(self, messages, **kwargs):
            model_questions.append(messages[-1].content)
            return AIMessage(content="请补充主播信息")

    service = module.CommerceAssistantService(LLM(), tool_retriever=object(), knowledge_retriever=object())
    final = list(service.events(question, [], user=ANONYMOUS_USER))[-1]
    assert model_questions == [question]
    assert set(offered_tools) == {"resolve_anchor", "analyze_anchor_live"}
    assert final["query_details"]["analysis_type"] == "creator_commerce"
    assert final["query_details"]["status"] == "clarify"
    assert final["response_type"] == "clarify"
    assert final["evidence_sufficient"] is False
    assert "主播名称或主播ID" in final["answer"]
