"""Catalog of business tools and the RAG-backed discovery index."""
from __future__ import annotations

import hashlib
import json
import logging
import re

from langchain_core.documents import Document
from core.vector_store import get_vector_store_manager

logger = logging.getLogger(__name__)
TOOL_SPECS = {
    "resolve_anchor": {"title": "按名称或ID查找达人账号", "domain": "达人账号",
        "description": "把用户提到的达人名称或账号ID解析成唯一anchor_id；支持名称搜索。重名时返回候选项，不能猜选。",
        "tags": ["主播", "达人", "账号", "主播ID", "anchor_id", "账号名称"],
        "inputs": {"anchor_name": "达人名称或账号ID，用户未提供时必须先追问"},
        "output": "候选账号ID、名称、主营类目和匹配状态", "required": ["anchor_name"],
        "examples": ["查找陈洁KiKi对应的主播ID"]},
    "analyze_anchor_live": {"title": "分析达人直播表现", "domain": "达人直播",
        "description": "查询指定主播和日期范围内的直播销量、GMV、场次、活跃日、品类分布和头部商品。本机MySQL模式使用live_base、live_sales_info和live_product_sale等字典业务表，数据为模拟记录。",
        "tags": ["主播", "达人", "直播", "直播间", "场次", "直播销量", "GMV"],
        "inputs": {"anchor_id": "经账号查找工具确认的ID", "start_date": "含首日的YYYY-MM-DD", "end_date": "含末日的YYYY-MM-DD"},
        "output": "直播销量、GMV、场次、品类和商品统计，含时间范围及模拟数据标记",
        "required": ["anchor_id", "start_date", "end_date"], "examples": ["查询A001近90天直播销量和场次"]},
    "analyze_anchor_video": {"title": "分析达人视频带货表现", "domain": "达人视频",
        "description": "查询指定主播和日期范围内的视频销量、GMV、视频数、品类分布和头部商品。本机MySQL模式使用anchor_video_item和date_product_video_incr等字典业务表；没有播放量时明确返回缺失。",
        "tags": ["主播", "达人", "视频", "短视频", "作品", "视频销量", "播放量"],
        "inputs": {"anchor_id": "经账号查找工具确认的ID", "start_date": "含首日的YYYY-MM-DD", "end_date": "含末日的YYYY-MM-DD"},
        "output": "视频销量、GMV、视频数量、播放量、品类和商品统计，含时间范围及模拟数据标记",
        "required": ["anchor_id", "start_date", "end_date"], "examples": ["查询A001近90天视频销量和播放量"]},
    "analyze_anchor_products": {"title": "分析达人带货商品和品类", "domain": "达人商品选品",
        "description": "汇总指定主播在日期范围内的直播和视频商品销量、销售额、品类占比及头部商品集中度。本机MySQL模式查询product、live_product_sale和date_product_video_incr等字典业务表。",
        "tags": ["主播", "达人", "商品", "品类", "类目", "带货", "选品", "品牌", "价格带"],
        "inputs": {"anchor_id": "经账号查找工具确认的ID", "start_date": "含首日的YYYY-MM-DD", "end_date": "含末日的YYYY-MM-DD"},
        "output": "商品和品类销量、金额、占比、Top商品及集中度，含模拟数据标记",
        "required": ["anchor_id", "start_date", "end_date"], "examples": ["分析A001近90天带货商品和品类结构"]},
    "compare_anchor_category": {"title": "对比同类达人表现", "domain": "达人竞品对比",
        "description": "将指定达人与当前业务库中相同主营类目的其他达人对比指定日期范围销量；返回样本量，样本过少时明确提示。当前数据为本地模拟数据。",
        "tags": ["主播", "达人", "竞品", "同类", "对标", "行业", "比较", "选品", "类目"],
        "inputs": {"anchor_id": "经账号查找工具确认的ID", "start_date": "含首日的YYYY-MM-DD", "end_date": "含末日的YYYY-MM-DD"},
        "output": "本人总销量、同类账号销量中位数、对照账号及样本量警告，含模拟数据标记",
        "required": ["anchor_id", "start_date", "end_date"], "examples": ["比较A001近90天同类达人销量"]},
    "search_selection_knowledge": {"title": "检索选品与运营知识", "domain": "运营知识库",
        "description": "按已确认的主营品类、渠道特点和用户目标，从当前用户可访问且已发布的内部知识中找选品方法、运营案例和规则，返回片段与出处。",
        "tags": ["选品", "选品方向", "运营", "方法", "案例", "直播", "视频", "品类"],
        "inputs": {"query": "完整描述需要查找的品类、渠道和经营问题"},
        "output": "相关知识片段、原文件名、页码和文档ID；无资料时返回空结果", "required": ["query"],
        "examples": ["食品类目近90天直播选品建议与账号匹配方法"]},
}


def catalog_version():
    return hashlib.sha256(json.dumps(TOOL_SPECS, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def tool_document(tool_id):
    spec = TOOL_SPECS[tool_id]
    inputs = "；".join(f"{name}：{description}" for name, description in spec["inputs"].items())
    return (f"工具ID：{tool_id}\n工具名称：{spec['title']}\n所属领域：{spec['domain']}\n"
        f"用途：{spec['description']}\n适用关键词：{'、'.join(spec['tags'])}\n"
        f"必填参数：{'、'.join(spec['required'])}\n参数说明：{inputs}\n"
        f"返回内容：{spec['output']}\n调用示例：{'；'.join(spec['examples'])}")


def render_catalog_markdown():
    sections = ["# 对话式经营分析助手工具说明", "本文件由工具注册定义生成；工具检索索引写入现有RAG向量库。"]
    sections.extend(f"## {spec['title']}\n\n{tool_document(tool_id)}" for tool_id, spec in TOOL_SPECS.items())
    return "\n\n".join(sections)


def ensure_catalog_indexed(vector_store=None):
    manager = vector_store or get_vector_store_manager()
    version = catalog_version()
    current = manager.get_documents({"category": "assistant_tools", "catalog_version": version})
    indexed_ids = {metadata.get("tool_id") for metadata in current.get("metadatas", [])}
    missing = [tool_id for tool_id in TOOL_SPECS if tool_id not in indexed_ids]
    if not missing:
        return len(TOOL_SPECS)
    documents = [Document(page_content=tool_document(tool_id), metadata={
        "source": "init", "file_name": "assistant_tool_catalog.md", "original_filename": "assistant_tool_catalog.md",
        "category": "assistant_tools", "catalog_version": version, "tool_id": tool_id,
        "tenant_id": "default", "permission_scope": json.dumps({"public": True}, ensure_ascii=False),
    }) for tool_id in missing]
    if manager.add_documents(documents) != len(documents):
        logger.warning("助手工具说明未能完整写入RAG索引，将使用注册信息作为候选回退")
    return len(TOOL_SPECS)


def _token_set(value):
    return set(re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+", (value or "").lower()))


def discover_tool_ids(query, *, retriever=None, user=None, top_k=6):
    selected = []
    try:
        ensure_catalog_indexed(getattr(retriever, "vector_store", None))
        if retriever is not None:
            documents = retriever.retrieve(query, top_k=top_k,
                filter_dict={"category": "assistant_tools"}, user=user, apply_rerank=False)
            for document in documents:
                found_ids = []
                metadata_id = document.metadata.get("tool_id")
                if metadata_id in TOOL_SPECS:
                    found_ids.append(metadata_id)
                found_ids.extend(re.findall(r"工具ID\s*[:：]\s*([a-z][a-z0-9_]*)", document.page_content))
                for tool_id in found_ids:
                    if tool_id in TOOL_SPECS and tool_id not in selected:
                        selected.append(tool_id)
    except Exception as exc:
        logger.warning("RAG工具发现失败，回退到工具注册标签匹配：%s", type(exc).__name__)
    query_tokens = _token_set(query)
    fallback = []
    for tool_id, spec in TOOL_SPECS.items():
        tokens = _token_set(" ".join([spec["title"], spec["domain"], spec["description"], *spec["tags"]]))
        score = len(query_tokens & tokens)
        if score:
            fallback.append((score, tool_id))
    for _, tool_id in sorted(fallback, key=lambda item: (-item[0], item[1])):
        if tool_id not in selected:
            selected.append(tool_id)
    return selected[:top_k]
