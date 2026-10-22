"""验证模型主导的路由、输出校验和失败处理，不调用真实模型或业务数据库。"""
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from config.settings import settings
from core.intent_router import IntentPrediction, IntentRouter, IntentRoutingError


def prediction(route="knowledge_qa", intent="通用问答", **kwargs):
    return IntentPrediction(
        route=route, intent=intent,
        **{"confidence": 0.94, "need_clarification": False, **kwargs},
    )


class StructuredModel:
    def __init__(self, *responses, error=None):
        self.responses = responses
        self.error = error
        self.messages = []
        self.configs = []

    def with_structured_output(self, schema):
        assert schema is IntentPrediction
        return RunnableLambda(self.respond)

    def respond(self, messages, config=None):
        self.messages.append(messages)
        self.configs.append(config)
        if self.error:
            raise self.error
        return self.responses[min(len(self.messages) - 1, len(self.responses) - 1)]


class TextModel(StructuredModel):
    """模拟旧版 Ollama 等不支持原生结构化返回的客户端。"""

    def with_structured_output(self, schema):
        raise NotImplementedError

    def invoke(self, messages, config=None):
        return AIMessage(content=self.respond(messages, config))


@pytest.fixture(autouse=True)
def router_settings(monkeypatch):
    monkeypatch.setattr(settings, "INTENT_ROUTER_ENABLED", True)
    monkeypatch.setattr(settings, "INTENT_CONFIDENCE_THRESHOLD", 0.65)
    monkeypatch.setattr(settings, "SALES_QUERY_ENABLED", True)
    monkeypatch.setattr(settings, "COMMERCE_ASSISTANT_ENABLED", True)


@pytest.mark.parametrize(("question", "route", "intent"), [
    ("报销材料需要准备哪些？", "knowledge_qa", "政策咨询"),
    ("什么是直播销量？", "knowledge_qa", "知识查询"),
    ("如何提高直播销量？", "knowledge_qa", "操作指导"),
    ("直播和视频总销量前五主播", "sales_query", "销量查询"),
    ("上个月谁卖得最好？", "sales_query", "销量查询"),
    ("查询主播近30天直播销量排名", "sales_query", "销量查询"),
    ("帮我完整复盘 XX 店铺最新一场直播的数据，看看整体卖得怎么样", "commerce_analysis", "经营分析"),
])
def test_all_business_routes_follow_model_without_keyword_confirmation(monkeypatch, question, route, intent):
    def unexpected_rules(*args, **kwargs):
        pytest.fail("模型模式不能提前调用关键词分流，也不能要求关键词确认模型结果")

    monkeypatch.setattr(IntentRouter, "classify_by_rules", unexpected_rules)
    model = StructuredModel(prediction(route, intent))

    decision = IntentRouter(model).classify(question)

    assert decision.route == route
    assert decision.intent == intent
    assert decision.confidence == 0.94
    assert decision.source == "llm"
    assert len(model.messages) == 1


def test_followup_passes_recent_history_and_current_question_to_model():
    history = [HumanMessage(content="已过时的问题"), AIMessage(content="已过时的答案"),
        HumanMessage(content="查销量前五的主播"), AIMessage(content="第一名是陈洁"),
        HumanMessage(content="她上个月呢？"), AIMessage(content="上个月的销量"),
        HumanMessage(content="她的视频呢？"), AIMessage(content="视频销量")]
    model = StructuredModel(prediction("sales_query", "销量查询"))

    decision = IntentRouter(model).classify("那直播呢？", history, config={"configurable": {"request_id": "route-1"}})

    payload = json.loads(model.messages[0][-1].content)
    assert decision.route == "sales_query"
    assert payload["question"] == "那直播呢？"
    assert [item["content"] for item in payload["recent_history"]] == [m.content for m in history[-6:]]
    assert model.configs[0]["configurable"]["request_id"] == "route-1"


@pytest.mark.parametrize(("route", "need_clarification", "confidence"), [
    ("knowledge_qa", True, 0.94),
    ("clarification", False, 0.94),
    ("clarification", True, 0.2),
    ("sales_query", False, 0.2),
])
def test_ambiguity_and_low_confidence_ask_instead_of_guessing(route, need_clarification, confidence):
    model = StructuredModel(prediction(route, need_clarification=need_clarification,
        confidence=confidence, clarification_message="你想查询哪位主播？"))

    decision = IntentRouter(model).classify("那他呢？")

    assert decision.route == "clarification"
    assert decision.need_clarification
    assert decision.intent == "待澄清"
    assert decision.clarification_message == "你想查询哪位主播？"


def test_whitespace_clarification_gets_a_usable_default():
    model = StructuredModel(prediction("clarification", clarification_message="  "))
    decision = IntentRouter(model).classify("那个")
    assert decision.clarification_message.strip()


@pytest.mark.parametrize(("route", "name"), [("sales_query", "销量查询"), ("commerce_analysis", "经营分析")])
def test_program_normalizes_business_labels(route, name):
    model = StructuredModel(prediction(route, "随意的名称", clarification_message="不需要的追问"))
    decision = IntentRouter(model).classify("具体业务问题")
    assert decision.intent == name
    assert decision.clarification_message is None


@pytest.mark.parametrize(("route", "setting", "message"), [
    ("sales_query", "SALES_QUERY_ENABLED", "销量查询功能当前未开启"),
    ("commerce_analysis", "COMMERCE_ASSISTANT_ENABLED", "经营分析功能当前未开启"),
])
def test_model_cannot_enable_disabled_business(monkeypatch, route, setting, message):
    monkeypatch.setattr(settings, setting, False)
    model = StructuredModel(prediction(route))
    with pytest.raises(IntentRoutingError, match=message):
        IntentRouter(model).classify("请查询业务数据")
    assert len(model.messages) == 1


@pytest.mark.parametrize("change", [
    {"route": "delete_database"}, {"confidence": 1.2}, {"sql": "SELECT 1"}, {"source": "rule"},
])
def test_invalid_model_fields_are_retried_once(change):
    invalid = {**prediction().model_dump(), **change}
    model = StructuredModel(invalid, prediction("sales_query", "销量查询"))

    decision = IntentRouter(model).classify("上个月谁卖得最好？")

    assert decision.route == "sales_query"
    assert decision.source == "llm"
    assert len(model.messages) == 2


def test_incomplete_model_fields_fail_after_one_format_retry():
    model = StructuredModel({"route": "knowledge_qa"})
    with pytest.raises(IntentRoutingError, match="稍后重试"):
        IntentRouter(model).classify("怎么申请年假？")
    assert len(model.messages) == 2


@pytest.mark.parametrize("error", [RuntimeError("private-provider-detail"), TimeoutError("private-timeout")])
def test_provider_failure_does_not_guess_using_keyword_rules(monkeypatch, error):
    def unexpected_rules(*args, **kwargs):
        pytest.fail("模型故障后不能静默退回关键词判断")

    monkeypatch.setattr(IntentRouter, "classify_by_rules", unexpected_rules)
    model = StructuredModel(error=error)
    with pytest.raises(IntentRoutingError, match="稍后重试") as exc:
        IntentRouter(model).classify("直播和视频总销量前五主播")
    assert "private" not in str(exc.value)
    assert len(model.messages) == 1


def test_missing_model_reports_unavailability():
    with pytest.raises(IntentRoutingError, match="稍后重试"):
        IntentRouter().classify("怎么申请年假？")


@pytest.mark.parametrize("fenced", [False, True])
def test_model_without_native_structured_output_still_classifies_semantically(fenced):
    payload = json.dumps(prediction("commerce_analysis", "经营分析").model_dump(), ensure_ascii=False)
    model = TextModel("```json\n" + payload + "\n```" if fenced else payload)

    decision = IntentRouter(model).classify("帮我完整复盘 XX 店铺最新一场直播的数据，看看整体卖得怎么样")

    assert decision.route == "commerce_analysis"
    assert decision.source == "llm"
    assert len(model.messages) == 1


def test_invalid_json_is_repaired_once_by_model():
    model = TextModel("这不是JSON", json.dumps(prediction("sales_query", "销量查询").model_dump()))
    decision = IntentRouter(model).classify("谁卖得最好？")
    assert decision.route == "sales_query"
    assert len(model.messages) == 2


def test_truncated_json_is_rejected_instead_of_silently_repaired():
    payload = json.dumps(prediction("sales_query", "销量查询").model_dump())
    model = TextModel(payload[:-1])
    with pytest.raises(IntentRoutingError, match="稍后重试"):
        IntentRouter(model).classify("谁卖得最好？")
    assert len(model.messages) == 2


def test_model_without_structured_output_method_can_return_json():
    model = RunnableLambda(lambda messages: AIMessage(content=json.dumps(prediction().model_dump())))
    assert IntentRouter(model).classify("什么是年假？").route == "knowledge_qa"


def test_legacy_rules_require_explicitly_disabling_model_router(monkeypatch):
    monkeypatch.setattr(settings, "INTENT_ROUTER_ENABLED", False)
    model = StructuredModel(error=AssertionError("旧模式不调用模型"))

    decision = IntentRouter(model).classify("直播和视频总销量前五主播")

    assert decision.route == "sales_query"
    assert decision.source == "rule"
    assert model.messages == []
