from datetime import date, timedelta
import json

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from core.auth import ANONYMOUS_USER
from core.assistant_tool_catalog import TOOL_SPECS, catalog_version, discover_tool_ids, tool_document
from core.commerce_assistant import CommerceAssistantService, _period_from_question, is_commerce_analysis_question
from core.commerce_demo_data import (
    analyze_anchor_live, analyze_anchor_products, analyze_anchor_video,
    calculate_channel_metrics, compare_anchor_category, demo_data_summary, find_anchors, seed_demo_commerce_data,
)


def test_seed_creates_isolated_repeatable_90_day_data(tmp_path):
    path = tmp_path / "demo.sqlite3"
    end = date(2026, 10, 17)
    first = seed_demo_commerce_data(path, end_date=end)
    second = seed_demo_commerce_data(path, end_date=end)
    assert first["dataset_label"] == second["dataset_label"] == "SIMULATED"
    assert (first["anchors"], first["products"]) == (6, 12)
    assert first["live_rows"] > 0 and first["video_rows"] > 0
    assert (first["period_start"], first["period_end"]) == ("2026-07-20", "2026-10-17")
    assert second["live_rows"] == first["live_rows"]
    assert len(find_anchors("陈洁KiKi", path)) == 1


def test_demo_queries_return_consistent_metrics_and_daily_window(tmp_path):
    path = tmp_path / "demo.sqlite3"
    seed_demo_commerce_data(path, end_date=date(2026, 10, 17))
    live = analyze_anchor_live("A001", "2026-07-20", "2026-10-17", path)
    video = analyze_anchor_video("A001", "2026-07-20", "2026-10-17", path)
    products = analyze_anchor_products("A001", "2026-07-20", "2026-10-17", path)
    metrics = calculate_channel_metrics(live, video, products)
    comparison = compare_anchor_category("A001", "2026-07-20", "2026-10-17", path)
    assert comparison["peer_count"] == 1
    assert comparison["sample_size_warning"]
    assert live["dataset_label"] == video["dataset_label"] == products["dataset_label"] == "SIMULATED"
    assert live["period_start"] == video["period_start"] == products["period_start"] == "2026-07-20"
    assert live["sales_quantity"] > 0 and video["sales_quantity"] > 0
    assert sum(item["sales_quantity"] for item in products["categories"]) == products["total_sales_quantity"]
    assert 0 < metrics["live_sales_share"] < 1
    assert round(metrics["live_sales_share"] + metrics["video_sales_share"], 6) == 1
    partial = calculate_channel_metrics(live, None, products)
    assert partial["live_sales_share"] is None and partial["video_sales_share"] is None
    assert partial["comparison_status"] == "incomplete_channel_data"
    import pytest
    with pytest.raises(ValueError, match="仅覆盖"):
        analyze_anchor_live("A001", "2026-07-19", "2026-10-17", path)


def test_date_period_parser_and_analysis_routing():
    start, end, label = _period_from_question("分析陈洁KiKi近90天表现", date(2026, 10, 17))
    assert (start, end) == ("2026-07-20", "2026-10-17")
    assert "90天" in label
    assert is_commerce_analysis_question("分析这个主播近90天直播和视频表现，并给选品方向")
    assert is_commerce_analysis_question("接下来选品方向", [HumanMessage(content="主播陈洁KiKi近90天直播表现")])
    assert not is_commerce_analysis_question("你好")
    assert not is_commerce_analysis_question("谢谢", [HumanMessage(content="主播陈洁KiKi近90天直播表现")])
    assert is_commerce_analysis_question("那和同行比呢", [HumanMessage(content="主播陈洁KiKi近90天直播表现")])
    assert is_commerce_analysis_question("查一下主播 A001 本月直播和视频销量分别是多少？")
    assert is_commerce_analysis_question("分析陈洁KiKi近90天直播带货数据并给出选品方向")
    assert is_commerce_analysis_question("这个主播近90天直播和视频表现", [HumanMessage(content="我的账号是陈洁KiKi")])
    assert not is_commerce_analysis_question("查一下主播销量")
    sales_history = [HumanMessage(content="直播和视频总销量前五主播"),
        AIMessage(content="查询结果", additional_kwargs={"query_details": {"status": "succeeded"}})]
    assert not is_commerce_analysis_question("他的视频销量呢？", sales_history)
    assert not is_commerce_analysis_question("查询店铺商品销量排名")
    assert not is_commerce_analysis_question("什么是直播销量？")


def test_tool_docs_are_searchable_and_keep_executable_ids():
    class Store:
        def __init__(self):
            self.docs = []
        def get_documents(self, filters=None):
            return {"documents": [d.page_content for d in self.docs], "metadatas": [d.metadata for d in self.docs]}
        def add_documents(self, docs):
            self.docs.extend(docs)
            return len(docs)

    class Retriever:
        def __init__(self):
            self.vector_store = Store()
        def retrieve(self, query, **kwargs):
            filters = kwargs.get("filter_dict", {})
            return [Document(page_content=doc.page_content,
                    metadata={key: value for key, value in doc.metadata.items() if key != "tool_id"})
                for doc in self.vector_store.docs if doc.metadata.get("category") == filters.get("category")]

    retriever = Retriever()
    candidates = discover_tool_ids("主播近90天直播视频销量和选品方向", retriever=retriever, user=ANONYMOUS_USER)
    assert {"analyze_anchor_live", "analyze_anchor_video", "analyze_anchor_products", "search_selection_knowledge"} <= set(candidates)
    assert all(item.metadata.get("tool_id") in TOOL_SPECS for item in retriever.vector_store.docs)
    assert all(item.metadata.get("catalog_version") == catalog_version() for item in retriever.vector_store.docs)
    assert "工具ID：analyze_anchor_live" in tool_document("analyze_anchor_live")


class ScriptedToolLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.bound_names = []
    def bind_tools(self, tools):
        self.bound_names.append([tool.name for tool in tools])
        return self
    def invoke(self, messages, config=None):
        if not self.responses:
            raise AssertionError("no scripted model response left")
        return self.responses.pop(0)


class ToolCatalogRetriever:
    def __init__(self):
        class Store:
            def get_documents(self, *args, **kwargs):
                return {"documents": [], "metadatas": []}
            def add_documents(self, docs):
                return len(docs)
        self.vector_store = Store()
    def retrieve(self, query, **kwargs):
        return [Document(page_content=tool_document(tool_id), metadata={"tool_id": tool_id})
            for tool_id in TOOL_SPECS]


class KnowledgeRetriever:
    def retrieve(self, query, **kwargs):
        return [Document(page_content="选品先围绕已验证的主力类目小批测试，再观察渠道表现。",
            metadata={"original_filename": "选品方法.md", "page": 2, "document_id": "doc-1", "version_id": "v1"})]


def test_agent_resolves_account_parallel_queries_retrieves_knowledge_and_summarizes(tmp_path, monkeypatch):
    from config.settings import settings
    demo_path = tmp_path / "demo.sqlite3"
    monkeypatch.setattr(settings, "COMMERCE_DEMO_DATABASE_PATH", demo_path)
    seed_demo_commerce_data(demo_path, end_date=date.today())
    monkeypatch.setattr("core.commerce_assistant.discover_tool_ids", lambda *args, **kwargs: list(TOOL_SPECS))
    responses = [
        AIMessage(content="", tool_calls=[{"name": "resolve_anchor", "args": {"anchor_name": "陈洁KiKi"}, "id": "c1", "type": "tool_call"}]),
        AIMessage(content="", tool_calls=[
            {"name": "analyze_anchor_live", "args": {"anchor_id": "A001", "start_date": "2026-07-20", "end_date": "2026-10-17"}, "id": "c2", "type": "tool_call"},
            {"name": "analyze_anchor_video", "args": {"anchor_id": "A001", "start_date": "2026-07-20", "end_date": "2026-10-17"}, "id": "c3", "type": "tool_call"},
            {"name": "analyze_anchor_products", "args": {"anchor_id": "A001", "start_date": "2026-07-20", "end_date": "2026-10-17"}, "id": "c4", "type": "tool_call"},
        ]),
        AIMessage(content="数据查询完成"),
        AIMessage(content="", tool_calls=[{"name": "search_selection_knowledge", "args": {"query": "美妆个护类目近90天直播和视频选品方向"}, "id": "c5", "type": "tool_call"}]),
        AIMessage(content="这是一份基于模拟数据的报告。"),
    ]
    llm = ScriptedToolLLM(responses)
    service = CommerceAssistantService(llm, tool_retriever=ToolCatalogRetriever(), knowledge_retriever=KnowledgeRetriever())
    events = list(service.events("分析陈洁KiKi近90天直播和视频表现，并给出选品方向。", [], user=ANONYMOUS_USER))
    result = events[-1]
    details = result["query_details"]
    assert result["response_type"] == "analysis_report"
    assert details["account"]["anchor_id"] == "A001"
    expected_end = date.today()
    expected_start = expected_end - timedelta(days=89)
    assert details["period_start"] == expected_start.isoformat() and details["period_end"] == expected_end.isoformat()
    assert set(details["tools_used"]) >= {"resolve_anchor", "analyze_anchor_live", "analyze_anchor_video", "analyze_anchor_products", "search_selection_knowledge"}
    assert details["calculated_metrics"]["dataset_label"] == "SIMULATED"
    assert result["context"][0].metadata["document_id"] == "doc-1"
    assert "search_selection_knowledge" in llm.bound_names[-1]


def test_rag_chain_routes_complex_business_question_to_analysis_service(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "AGENT_ENABLED", False)
    from types import SimpleNamespace
    from core import rag_chain as rag_module
    from core.auth import ANONYMOUS_USER
    from core.intent_recognizer import IntentRecognizer
    from core.memory_manager import MemoryManager
    from core.rag_chain import RAGChain

    called = []
    class StubAssistant:
        def __init__(self, *args, **kwargs):
            pass
        def events(self, question, history, **kwargs):
            called.append(question)
            yield {"answer": "演示报告", "context": [], "query_details": {"analysis_type": "creator_commerce"},
                "evidence_sufficient": True, "response_type": "analysis_report", "citation_status": "database"}

    monkeypatch.setattr(rag_module, "CommerceAssistantService", StubAssistant)
    chain = object.__new__(RAGChain)
    # 此测试只验证经营分析分发；模型分类和故障兜底由 test_intent_router.py 单独验证。
    from core.intent_router import IntentDecision
    chain.intent_router = SimpleNamespace(classify=lambda *args, **kwargs: IntentDecision(
        route="commerce_analysis", intent="经营分析", confidence=0.95,
        need_clarification=False, source="llm"))
    chain.llm = None
    chain.intent_recognizer = IntentRecognizer()
    chain.memory_manager = MemoryManager()
    chain.retriever = SimpleNamespace()
    chain._rag_chain = None
    result = chain.ask("分析陈洁KiKi近90天直播带货数据并给出选品方向", "route-test",
        user=ANONYMOUS_USER, history=[], persist=False)
    assert called == ["分析陈洁KiKi近90天直播带货数据并给出选品方向"]
    assert result["intent"] == "经营分析"
    assert result["response_type"] == "analysis_report"
