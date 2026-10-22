"""Controlled agent flow for synthetic creator commerce analysis."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import json
import logging
import re
import threading

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from core.assistant_tool_catalog import TOOL_SPECS, discover_tool_ids
from core.commerce_data import (
    analyze_anchor_live, analyze_anchor_products, analyze_anchor_video,
    calculate_channel_metrics, compare_anchor_category, find_anchor_mentions, find_anchors,
)
from core.retriever import get_rag_retriever

logger = logging.getLogger(__name__)
MAX_TOOL_ROUNDS = 5
MAX_TOOL_CALLS = 10
MAX_TOOL_RESULT_CHARS = 14000
DATA_TOOLS = {"analyze_anchor_live", "analyze_anchor_video", "analyze_anchor_products", "compare_anchor_category"}


class AnchorLookupArgs(BaseModel):
    anchor_name: str = Field(min_length=1, max_length=128, description="主播名称或账号ID；不能猜测")


class AnchorPeriodArgs(BaseModel):
    anchor_id: str = Field(min_length=1, max_length=64, description="已经确认的主播ID")
    start_date: str = Field(description="YYYY-MM-DD，含当天")
    end_date: str = Field(description="YYYY-MM-DD，含当天")


class KnowledgeSearchArgs(BaseModel):
    query: str = Field(min_length=3, max_length=600, description="品类、渠道和经营目标组成的检索问题")


def is_commerce_analysis_question(question: str, history=None) -> bool:
    value = (question or "").strip()
    if not value or re.search(r"什么是|是什么意思|定义|计算公式|计算方法", value):
        return False
    history_text = "\n".join(str(getattr(message, "content", "")) for message in (history or [])[-6:])
    combined = value + "\n" + history_text
    business_topic = bool(re.search(r"直播|视频|短视频|带货|销量|销售额|GMV|选品|品类|商品", combined, re.I))
    creator_context = bool(re.search(r"主播|达人|账号|我的账号|我家|店铺|旗舰店|专营店|专卖店", combined))
    has_period = bool(re.search(r"近\s*(?:7|30|60|90|120|七|三十|六十|九十|一百二十)\s*(?:天|日|周|个月|月)|本月|这个月|上个月|上月|最新一场|最近一场|最后一场", value))
    analysis = bool(re.search(r"分析|表现|选品|建议|方向|诊断|复盘|怎么做|推荐|对比|竞品|同类|对标|同行", value, re.I))
    contextual_followup = bool(history_text and re.match(r"^(?:那|他|她|它|这个账号|这个主播|该主播|接下来)", value)
        and analysis and re.search(r"直播|视频|带货|选品|品类", combined))
    return (business_topic and (analysis or has_period) and (creator_context or has_period)) or contextual_followup


def _period_from_question(text: str, today: date | None = None) -> tuple[str, str, str]:
    today = today or date.today()
    value = (text or "").replace(" ", "")
    if re.search(r"本月|这个月|当月", value):
        return today.replace(day=1).isoformat(), today.isoformat(), "本月截至今天"
    if re.search(r"上个月|上月", value):
        last_day = today.replace(day=1) - timedelta(days=1)
        return last_day.replace(day=1).isoformat(), last_day.isoformat(), "上一个自然月"
    absolute_dates = re.findall(r"\d{4}-\d{1,2}-\d{1,2}", value)
    if len(absolute_dates) >= 2:
        start, end = date.fromisoformat(absolute_dates[0]), date.fromisoformat(absolute_dates[1])
        if start <= end:
            return start.isoformat(), end.isoformat(), "用户指定日期范围"
    if len(absolute_dates) == 1:
        end = date.fromisoformat(absolute_dates[0])
        nearby_days = re.search(r"(\d+)\s*(?:天|日)", value)
        if nearby_days:
            days = max(1, min(int(nearby_days.group(1)), 120))
            return (end - timedelta(days=days - 1)).isoformat(), end.isoformat(), f"截至指定日期近{days}天"
        if re.search(r"(?:当天|当日|日期|这天|该日)", value):
            return end.isoformat(), end.isoformat(), "用户指定日期"
    matches = list(re.finditer(r"(?:近|最近|过去|前)?(\d+|[一二两三四五六七八九十百]+)(天|日|周|星期|个?月)", value))
    number = None
    unit = None
    if matches:
        number_text, unit = matches[-1].groups()
        chinese = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "百": 100}
        number = int(number_text) if number_text.isdigit() else chinese.get(number_text, 90)
    if number is None:
        days, assumption = 30, "用户未指定周期，按近30天处理"
    else:
        days, assumption = number * (7 if unit in {"周", "星期"} else 30 if "月" in unit else 1), f"近{number * (7 if unit in {"周", "星期"} else 30 if "月" in unit else 1)}天（含首尾日期）"
    days = max(1, min(days, 120))
    return (today - timedelta(days=days - 1)).isoformat(), today.isoformat(), assumption


def _serialize(value) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = json.dumps({"status": "error", "message": "工具结果无法格式化"}, ensure_ascii=False)
    return text[:MAX_TOOL_RESULT_CHARS]


def _source_documents(items: list[dict]):
    from langchain_core.documents import Document
    return [Document(page_content=item["content"], metadata={
        "original_filename": item.get("source", "内部知识库"), "file_name": item.get("source", "内部知识库"),
        "page": item.get("page"), "document_id": item.get("document_id"),
        "version_id": item.get("version_id"), "category": "knowledge",
    }) for item in items]


class CommerceAssistantService:
    def __init__(self, llm, *, tool_retriever=None, knowledge_retriever=None):
        self.llm = llm
        self.tool_retriever = tool_retriever or get_rag_retriever()
        self.knowledge_retriever = knowledge_retriever or self.tool_retriever

    def _knowledge_search(self, query, user):
        try:
            docs = self.knowledge_retriever.retrieve(query, top_k=5, user=user, apply_rerank=True)
        except Exception as exc:
            logger.warning("分析知识检索失败：%s", type(exc).__name__)
            return {"status": "unavailable", "items": [], "message": "内部知识检索暂不可用"}
        items = []
        for doc in docs:
            if doc.metadata.get("category") == "assistant_tools":
                continue
            items.append({"content": doc.page_content[:4500],
                "source": doc.metadata.get("original_filename") or doc.metadata.get("file_name") or doc.metadata.get("source") or "内部知识库",
                "page": doc.metadata.get("page"), "document_id": doc.metadata.get("document_id"),
                "version_id": doc.metadata.get("version_id")})
        return {"status": "succeeded" if items else "no_results", "items": items,
            "message": "已返回相关资料片段和出处" if items else "没有找到可访问的内部选品资料"}

    def events(self, question, history, *, user, config=None, cancelled=None):
        used_tools, results = [], {}
        plan_text = question + "\n" + "\n".join(f"{m.type}: {m.content}" for m in history[-8:])
        start_date, end_date, period_assumption = _period_from_question(plan_text)
        yield {"stage": "正在从工具说明库查找相关分析能力"}
        tool_ids = discover_tool_ids(plan_text, retriever=self.tool_retriever, user=user, top_k=8)
        if re.search(r"主播|达人|账号", plan_text):
            tool_ids.insert(0, "resolve_anchor")
        if re.search(r"直播", question):
            tool_ids.append("analyze_anchor_live")
        if re.search(r"视频|短视频|作品", question):
            tool_ids.append("analyze_anchor_video")
        if re.search(r"商品|品类|类目|选品|建议|方向", question):
            tool_ids.append("analyze_anchor_products")
        if re.search(r"竞品|同类|对标|行业|选品|建议|方向", question):
            tool_ids.append("compare_anchor_category")
        if re.search(r"选品|建议|方向|怎么做|推荐", question):
            tool_ids.append("search_selection_knowledge")
        tool_ids = list(dict.fromkeys(tool_id for tool_id in tool_ids if tool_id in TOOL_SPECS))
        account = {}
        data_tool_ids = [x for x in tool_ids if x in DATA_TOOLS or x == "resolve_anchor"]
        tool_functions = {}

        def resolve_anchor_tool(anchor_name: str):
            matches = find_anchors(anchor_name)
            if not matches:
                return {"status": "not_found", "message": "没有找到匹配账号，请核对主播名称或ID。"}
            if len(matches) > 1:
                return {"status": "ambiguous", "candidates": matches, "message": "找到多个相近账号，请让用户选择。"}
            account.update(matches[0])
            return {"status": "succeeded", "account": matches[0], "dataset_label": "SIMULATED"}

        def run_live(anchor_id: str, start_date: str, end_date: str):
            return analyze_anchor_live(account.get("anchor_id", anchor_id), start_date, end_date)

        def run_video(anchor_id: str, start_date: str, end_date: str):
            return analyze_anchor_video(account.get("anchor_id", anchor_id), start_date, end_date)

        def run_products(anchor_id: str, start_date: str, end_date: str):
            return analyze_anchor_products(account.get("anchor_id", anchor_id), start_date, end_date)

        def run_peer_comparison(anchor_id: str, start_date: str, end_date: str):
            return compare_anchor_category(account.get("anchor_id", anchor_id), start_date, end_date)

        tool_functions.update({
            "resolve_anchor": (resolve_anchor_tool, AnchorLookupArgs),
            "analyze_anchor_live": (run_live, AnchorPeriodArgs),
            "analyze_anchor_video": (run_video, AnchorPeriodArgs),
            "analyze_anchor_products": (run_products, AnchorPeriodArgs),
            "compare_anchor_category": (run_peer_comparison, AnchorPeriodArgs),
        })
        tools = [StructuredTool.from_function(func=tool_functions[tool_id][0], name=tool_id,
            description=TOOL_SPECS[tool_id]["description"], args_schema=tool_functions[tool_id][1])
            for tool_id in data_tool_ids]
        tool_map = {tool.name: tool for tool in tools}
        system = ("你是经营分析执行助手。只调用本轮提供的工具；数据库结果是唯一数值依据。"
            "先确认主播账号，找不到或重名时停止并澄清。独立的直播、视频、商品查询可以同轮发起。"
            "不得伪造账号ID、日期、数字或竞品结论；SIMULATED数据必须明确标注为模拟数据。")
        messages = [SystemMessage(content=system), *history[-8:], HumanMessage(content=question)]
        bound = None
        binding_available = bool(tools)
        try:
            if tools:
                bound = self.llm.bind_tools(tools)
        except Exception as exc:
            binding_available = False
            logger.info("当前模型不支持工具调用，将使用受控业务查询：%s", type(exc).__name__)
        total_calls = 0
        final_response = None
        yield {"stage": "正在确认账号并规划数据查询"}
        try:
            for _ in range(MAX_TOOL_ROUNDS if binding_available else 0):
                if cancelled is not None and cancelled.is_set():
                    return
                try:
                    response = bound.invoke(messages, config=config)
                except Exception as exc:
                    logger.info("模型工具调用失败，改用受控业务查询：%s", type(exc).__name__)
                    binding_available = False
                    break
                messages.append(response)
                calls = list(getattr(response, "tool_calls", []) or [])
                if not calls:
                    final_response = response
                    break
                if total_calls + len(calls) > MAX_TOOL_CALLS:
                    raise ValueError("本次分析的工具调用次数已达上限")
                total_calls += len(calls)
                if any(call.get("name") not in tool_map for call in calls):
                    raise ValueError("模型请求了本轮未授权的分析工具")

                def execute(call):
                    name, args = call["name"], dict(call.get("args") or {})
                    if name in DATA_TOOLS:
                        if not account:
                            provided = str(args.get("anchor_id", "")).strip()
                            matches = find_anchors(provided) if provided else []
                            if len(matches) == 1:
                                account.update(matches[0])
                        if not account:
                            return call, {"status": "need_anchor", "message": "必须先确认主播ID。"}
                        args["anchor_id"] = account["anchor_id"]
                        args["start_date"], args["end_date"] = start_date, end_date
                    try:
                        return call, tool_map[name].invoke(args)
                    except Exception as exc:
                        return call, {"status": "error", "message": str(exc)[:300]}

                resolve_calls = [c for c in calls if c["name"] == "resolve_anchor"]
                data_calls = [c for c in calls if c["name"] in DATA_TOOLS]
                completed = [execute(c) for c in resolve_calls]
                if len(data_calls) > 1:
                    yield {"stage": "正在并发查询直播、视频和商品数据"}
                    with ThreadPoolExecutor(max_workers=min(3, len(data_calls))) as pool:
                        completed.extend(pool.map(execute, data_calls))
                else:
                    completed.extend(execute(c) for c in data_calls)
                for call, value in completed:
                    results[call["name"]] = value
                    if call["name"] not in used_tools:
                        used_tools.append(call["name"])
                    messages.append(ToolMessage(content=_serialize(value), tool_call_id=call["id"]))

            mentions = find_anchor_mentions(plan_text)
            if not account and len(mentions) == 1:
                account.update(mentions[0])
                results.setdefault("resolve_anchor", {"status": "succeeded", "account": mentions[0],
                    "dataset_label": "SIMULATED", "resolved_from": "conversation_context"})
                used_tools.append("resolve_anchor")
            if account:
                requested = []
                if "直播" in question and "analyze_anchor_live" in tool_ids:
                    requested.append("analyze_anchor_live")
                if re.search(r"视频|短视频|作品", question) and "analyze_anchor_video" in tool_ids:
                    requested.append("analyze_anchor_video")
                if re.search(r"商品|品类|类目|选品|建议|方向", question) and "analyze_anchor_products" in tool_ids:
                    requested.append("analyze_anchor_products")
                if re.search(r"竞品|同类|对标|行业|选品|建议|方向", question) and "compare_anchor_category" in tool_ids:
                    requested.append("compare_anchor_category")
                runners = {"analyze_anchor_live": run_live, "analyze_anchor_video": run_video,
                    "analyze_anchor_products": run_products, "compare_anchor_category": run_peer_comparison}
                for tool_id in requested:
                    if not results.get(tool_id) or results[tool_id].get("status") in {"error", "need_anchor"}:
                        yield {"stage": "正在补齐" + TOOL_SPECS[tool_id]["title"] + "数据"}
                        results[tool_id] = runners[tool_id](account["anchor_id"], start_date, end_date)
                        if tool_id not in used_tools:
                            used_tools.append(tool_id)
            if final_response is None and binding_available:
                final_response = bound.invoke(messages, config=config)
            if cancelled is not None and cancelled.is_set():
                return

            if not account:
                resolution = results.get("resolve_anchor") or {}
                answer = resolution.get("message") or "请提供主播名称或主播ID，我再为你查询。"
                yield {"answer": answer, "context": [], "evidence_sufficient": False,
                    "response_type": "clarify", "citation_status": "not_applicable",
                    "query_details": {"status": "clarify", "analysis_type": "creator_commerce",
                        "tools_discovered": tool_ids, "tools_used": used_tools, "dataset_label": "SIMULATED"}}
                return

            live_result = results.get("analyze_anchor_live")
            video_result = results.get("analyze_anchor_video")
            product_result = results.get("analyze_anchor_products")
            live_result = live_result if live_result and live_result.get("status") != "error" else None
            video_result = video_result if video_result and video_result.get("status") != "error" else None
            product_result = product_result if product_result and product_result.get("status") != "error" else None
            if not any((live_result, video_result, product_result)):
                errors = [result.get("message") for result in results.values()
                    if isinstance(result, dict) and result.get("status") == "error" and result.get("message")]
                yield {"answer": errors[0] if errors else "没有查到可用于分析的数据，请检查账号和日期范围。", "context": [],
                    "query_details": {"status": "failed", "analysis_type": "creator_commerce", "dataset_label": "SIMULATED",
                        "period_start": start_date, "period_end": end_date, "tools_discovered": tool_ids,
                        "tools_used": used_tools, "results": results},
                    "evidence_sufficient": False, "response_type": "clarify", "citation_status": "not_applicable"}
                return
            knowledge_items = []
            if re.search(r"选品|建议|方向|怎么做|推荐", question) and "search_selection_knowledge" in tool_ids:
                categories = []
                for key in DATA_TOOLS:
                    categories.extend(item.get("category") for item in (results.get(key) or {}).get("categories", []) if item.get("category"))
                category_text = "、".join(dict.fromkeys(categories)) or account.get("main_category", "")
                default_query = f"{category_text}类目，结合达人近{max(1, (date.fromisoformat(end_date) - date.fromisoformat(start_date)).days + 1)}天直播和视频表现，选品方向、适用条件和验证方法"
                yield {"stage": "正在结合实际品类检索选品资料"}
                knowledge_query = default_query
                knowledge_tool = StructuredTool.from_function(
                    func=lambda query: self._knowledge_search(query, user), name="search_selection_knowledge",
                    description=TOOL_SPECS["search_selection_knowledge"]["description"], args_schema=KnowledgeSearchArgs)
                if binding_available:
                    try:
                        kb_model = self.llm.bind_tools([knowledge_tool])
                        kb_response = kb_model.invoke([SystemMessage(content="根据输入形成检索问题并调用search_selection_knowledge。"),
                            HumanMessage(content=_serialize({"question": question, "default_query": default_query,
                                "categories": categories, "period": [start_date, end_date]}))], config=config)
                        kb_call = next((c for c in getattr(kb_response, "tool_calls", []) or [] if c.get("name") == "search_selection_knowledge"), None)
                        if kb_call:
                            knowledge_query = kb_call.get("args", {}).get("query") or default_query
                    except Exception as exc:
                        logger.info("知识检索问题改写失败，使用程序生成的问题：%s", type(exc).__name__)
                results["search_selection_knowledge"] = self._knowledge_search(knowledge_query, user)
                knowledge_items = results["search_selection_knowledge"].get("items", [])
                used_tools.append("search_selection_knowledge")

            source_documents = _source_documents(knowledge_items)
            metrics = calculate_channel_metrics(live_result, video_result, product_result)
            payload = {"user_request": question, "account": account,
                "period": {"start": start_date, "end": end_date, "assumption": period_assumption},
                "data_results": {key: value for key, value in results.items() if key != "search_selection_knowledge"},
                "calculated_metrics": metrics, "knowledge_results": results.get("search_selection_knowledge", {"status": "not_requested", "items": []}),
                "limitations": ["经营数据是本地合成的模拟数据，不代表任何真实达人或店铺。", "模拟库没有退款、真实利润、转化归因或外部行业趋势数据。"]}
            yield {"stage": "正在核对数据依据并生成分析报告"}
            prompt = ("根据以下结构化结果生成中文经营分析报告。程序计算的数字不可改写；区分数据事实、判断、建议和待验证假设。"
                "引用知识时写出来源；没有知识资料要明确说明；不得把模拟数据描述为真实数据。"
                "按整体表现、直播与视频对比、商品与品类、选品方向、数据限制组织报告。\n" + _serialize(payload))
            report = self.llm.invoke([SystemMessage(content=prompt), HumanMessage(content="请输出最终报告。")], config=config)
            answer = str(getattr(report, "content", "") or getattr(final_response, "content", "") or "未能生成分析报告。")
            details = {"status": "succeeded", "analysis_type": "creator_commerce", "dataset_label": "SIMULATED",
                "period_start": start_date, "period_end": end_date, "period_assumption": period_assumption,
                "account": account, "tools_discovered": tool_ids, "tools_used": used_tools,
                "results": results, "calculated_metrics": metrics, "limitations": payload["limitations"],
                "knowledge_sources": [{key: item.get(key) for key in ("source", "page", "document_id", "version_id")} for item in knowledge_items]}
            yield {"answer": answer, "context": source_documents, "query_details": details,
                "evidence_sufficient": bool(live_result or video_result or product_result),
                "response_type": "analysis_report", "citation_status": "database_and_knowledge" if knowledge_items else "database"}
        except Exception as exc:
            logger.warning("经营分析未完成：%s", type(exc).__name__)
            yield {"answer": "经营分析暂时无法完成，请核对账号名称和日期范围后重试。", "context": [],
                "query_details": {"status": "failed", "analysis_type": "creator_commerce",
                    "tools_discovered": tool_ids, "tools_used": used_tools, "error": type(exc).__name__},
                "evidence_sufficient": False, "response_type": "error", "citation_status": "not_applicable"}
