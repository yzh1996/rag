"""Small, permission-checked capabilities for the conversation agent."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
import json
from typing import Any, Literal
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import inspect, text
from sqlalchemy.exc import DBAPIError

from config.settings import settings
from core.business_query import _storage_types_match, validate_business_sql
from core.business_reader import dataset_info, readonly_business_connection, scalar_value
from core.business_schema import business_metadata, catalog_hash, load_business_catalog, query_schema
from core.telemetry import span


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DiscoverArgs(Arguments):
    capabilities: list[Literal["business", "knowledge", "calculation"]] = Field(
        min_length=1, max_length=3, description="需要获取的能力，可选业务数据、内部知识或计算。")


class SchemaArgs(Arguments):
    tables: list[str] = Field(min_length=1, max_length=8, description="需要查看字段说明的业务表名，最多8张。")


class QueryArgs(SchemaArgs):
    sql: str = Field(min_length=1, max_length=16000,
        description="只读SELECT语句。查询中的每张表必须先读取字段说明。")


class KnowledgeArgs(Arguments):
    query: str = Field(min_length=1, max_length=2000, description="要在当前用户可访问的内部资料中查找的问题。")


class CellRef(Arguments):
    evidence_id: str = Field(description="查询或计算结果返回的依据编号。")
    row: int = Field(default=0, ge=0, le=99, description="结果行号，从0开始。")
    column: str = Field(description="结果中原样的列名。")


class CalculationArgs(Arguments):
    operation: Literal["sum", "difference", "ratio", "percent_change"] = Field(
        description="计算方式：求和、差值、比值或百分比变化。")
    references: list[CellRef] = Field(min_length=1, max_length=100,
        description="需要计算的结果单元格；差值、比值和变化率按本期、基准的顺序填写。")


class FinishArgs(Arguments):
    answer: str = Field(min_length=1, max_length=30000, description="给用户的最终回答或需要补充说明。")
    response_type: Literal["answer", "clarify", "refuse"] = Field(
        default="answer", description="回答、追问用户，或说明当前无法回答。")
    goal: str = Field(default="", max_length=1000, description="当前任务的简短目标，供后续追问沿用。")
    facts: dict[str, CellRef] = Field(default_factory=dict, max_length=30,
        description="已确认对象和时间；每项都要引用真实查询结果中的行和列。")
    reset_task: bool = Field(default=False, description="用户切换对象或时间条件时设为true，清除旧任务依据。")
    pending_questions: list[str] = Field(default_factory=list, max_length=5,
        description="尚需向用户确认的问题，最多5项。")


@dataclass
class AgentToolContext:
    """每轮对话独享的工具运行状态，不跨请求或用户共享。"""

    user: Any
    evidence: dict
    documents: list[Document]
    retriever: Any
    version_manifest: Any
    capabilities: dict[str, str]
    available: set[str]
    current_ids: set[str]
    read_tables: set[str]
    task_state: dict
    question: str
    cancelled: Any = None
    registry: dict[str, BaseTool] = field(default_factory=dict)
    final: FinishArgs | None = None


BUSINESS_RULES = """业务口径：当前本地经营数据为模拟数据，金额按人民币元、销量按件。
shop.shop_id=jl_user.shop_id；jl_user.anchor_id=live_base.anchor_id；live_base.id=直播明细表room_id。
店铺默认是自播账号；最新已结束直播按live_create_time DESC,id DESC，finish_time>0且is_show=0。
任何对象的最新/前N/排名，都先按该对象过滤，再排序和LIMIT；不能先取全库最新N条再关联对象。
SQL仅允许SUM COUNT AVG MIN MAX COALESCE NULLIF ROUND ABS LOWER UPPER LENGTH CAST DATE和CASE等已登记函数；名称模糊查询用LIKE。
直播销售额取live_sales_info.sales；直播件数取live_product_sale.number；各自先汇总再关联，避免一对多重复求和。
视频金额/销量取date_product_video_incr.sales/sales_volume，date_flag是YYYYMMDD整数。
product.sales是商品累计销量，不能当作单场销量；product.id是字符串，关联数字商品ID时用CAST。
不要把原始、加工和汇总层重复相加。不具备订单、退款、成本、利润、购买转化率和因果归因数据。
默认优先原始表；用户明确指定加工或汇总表时可按已确认字段查询，必须保持统计粒度一致。
先验证是否真的卖得不好，说明比较基准；相关变化不等于因果。
"""

CAPABILITIES = {
    "business": "查本地MySQL：按需读取表字段说明，再执行只读SQL；适用于店铺、主播、直播、商品和视频。",
    "knowledge": "检索当前用户可访问的已发布内部知识，返回片段和出处；适用于公司规定、运营方法、指标口径。",
    "calculation": "使用查询结果中的数值，计算差值、比值、变化率和总和。",
}


def ensure_business_access(user):
    if not settings.SALES_QUERY_ENABLED:
        raise ValueError("本地数据查询功能当前未开启")
    # This synthetic business database has no tenant/department columns.
    if user.tenant_id != "default":
        raise ValueError("当前业务库尚未配置该租户的数据授权，不能访问共享模拟数据")


def schema_catalog(user):
    ensure_business_access(user)
    known = query_schema()
    return [{"table": t["name"], "layer": t["layer"], "field_count": len(known[t["name"]])}
            for t in load_business_catalog()["tables"] if t["name"] in known]


def read_schema(tables, user):
    ensure_business_access(user)
    if not tables or len(tables) > 8 or set(tables) - query_schema().keys():
        raise ValueError("每次请选择1至8张具有已确认字段的字典业务表")
    schema = query_schema(set(tables))
    if set(tables) != set(schema):
        raise ValueError("所选表缺少可确认的字段定义")
    return {"catalog_hash": catalog_hash(), "rules": BUSINESS_RULES, "tables": [
        {"table": t["name"], "fields": [{"name": c["name"], "type": c["source_type"], "description": c["description"]}
            for c in t["columns"] if c["name"] in schema[t["name"]]]}
        for t in load_business_catalog()["tables"] if t["name"] in schema]}


def query_business(sql, tables, user, cancelled=None):
    ensure_business_access(user)
    if not tables or set(tables) - query_schema().keys():
        raise ValueError("只能查询已读取说明且字段已确认的字典业务表")
    schema = query_schema(set(tables))
    max_rows = max(1, min(settings.SALES_QUERY_MAX_ROWS, 100))
    checked, used = validate_business_sql(sql, schema, max_rows)
    with readonly_business_connection(cancelled=cancelled) as connection:
        info = dataset_info(connection)
        inspector = inspect(connection)
        expected = business_metadata()
        for name in used:
            actual = {c["name"]: c for c in inspector.get_columns(name)}
            if not schema[name].keys() <= actual.keys() or any(
                    not _storage_types_match(actual[c]["type"], expected.tables[name].c[c].type) for c in schema[name]):
                raise ValueError("实际业务表与字段字典不一致，请先核对初始化状态")
        # Let the dialect escape literal percent signs for PyMySQL (e.g. LIKE '%店名%').
        try:
            result = connection.execute(text(checked))
        except DBAPIError as exc:
            code = getattr(exc.orig, "args", (None,))[0]
            messages = {1235: "MySQL不支持该查询写法；IN子查询中不能直接使用LIMIT，请改用派生表JOIN或在外层排序限量",
                1064: "MySQL查询语法不正确，请依据已提供的字段说明修正查询",
                1055: "分组查询包含未聚合字段，请调整GROUP BY或聚合表达式",
                3024: "数据库查询超时，请缩小时间范围或减少关联数据",
                1142: "数据库账号没有读取该业务表的权限"}
            raise ValueError(messages.get(code, "数据库查询未完成，请核对表结构、连接状态或缩小查询范围")) from None
        columns = list(result.keys())
        if len(columns) != len(set(columns)):
            raise ValueError("查询结果列名重复，请使用不同别名")
        output, size, truncated = [], 0, False
        for record in result.fetchmany(max_rows + 1):
            record = [scalar_value(v) for v in record]
            size += len(json.dumps(record, ensure_ascii=False).encode("utf-8"))
            if len(output) >= max_rows or size > 24000:
                truncated = True
                break
            output.append(record)
    return {"status": "succeeded", "kind": "database", "sql": checked, "tables": used,
        "columns": columns, "rows": output, "row_count": len(output), "truncated": truncated,
        "dataset_label": info["dataset_label"], "data_through": info["end_date"],
        "catalog_hash": catalog_hash(), "checked_at": datetime.now(timezone.utc).isoformat()}


def cell_value(evidence, reference):
    item = evidence.get(reference.evidence_id)
    if not item or item.get("status") != "succeeded":
        raise ValueError("所引用的查询依据不存在或已失效")
    try:
        return item["rows"][reference.row][item["columns"].index(reference.column)]
    except (IndexError, KeyError, ValueError):
        raise ValueError("所引用的结果行或列不存在") from None


def calculate(operation, references, evidence):
    values = [Decimal(str(cell_value(evidence, r))) for r in references]
    if any(not v.is_finite() for v in values):
        raise ValueError("只能计算有限数值")
    if operation == "sum":
        value = sum(values)
    elif len(values) != 2:
        raise ValueError("差值、比值、变化率需要两个数，依次为本期和基准")
    elif operation == "difference":
        value = values[0] - values[1]
    elif values[1] == 0:
        return {"status": "succeeded", "kind": "calculation", "columns": ["value"], "rows": [[None]],
            "message": "基准为零，不能计算比值或变化率"}
    else:
        value = (values[0] / values[1] - (1 if operation == "percent_change" else 0)) * (
            100 if operation == "percent_change" else 1)
    return {"status": "succeeded", "kind": "calculation", "columns": ["value"],
        "rows": [[scalar_value(value.quantize(Decimal("0.000001")))]], "operation": operation,
        "inputs": [r.model_dump() for r in references]}


class BusinessTools:
    """业务表说明与只读查询。底层校验仍由现有数据访问函数负责。"""

    capability = "business"

    def __init__(self, context: AgentToolContext):
        self.context = context

    def get_tools(self) -> list[BaseTool]:
        # 在方法内定义工具，用闭包持有当前请求的身份，避免把self暴露成模型参数。
        context = self.context

        @tool(args_schema=SchemaArgs)
        def read_business_schema(tables: list[str]) -> dict:
            """读取指定业务表的已确认字段和统计口径，查询前必须读取所用表。"""
            value = read_schema(tables, context.user)
            context.read_tables.update(tables)
            return value

        @tool(args_schema=QueryArgs)
        def query_database(tables: list[str], sql: str) -> dict:
            """执行单条只读MySQL SELECT，返回有编号的真实查询结果；禁止写入，必须先读取所有所用表说明。"""
            if set(tables) - context.read_tables:
                raise ValueError("查询前必须读取所有所用表的字段说明")
            with span("rag.agent.query"):
                value = query_business(sql, tables, context.user, context.cancelled)
            key = "data_" + uuid4().hex[:12]
            value["evidence_id"] = key
            context.evidence[key] = value
            context.current_ids.add(key)
            return value

        return [read_business_schema, query_database]


class KnowledgeTools:
    """使用当前请求的身份和发布版本检索内部知识。"""

    capability = "knowledge"

    def __init__(self, context: AgentToolContext):
        self.context = context

    def get_tools(self) -> list[BaseTool]:
        context = self.context

        @tool(args_schema=KnowledgeArgs)
        def search_knowledge(query: str) -> dict:
            """检索当前用户可访问的内部知识与出处；每轮重新检查权限和发布版本。"""
            with span("rag.retrieval"):
                found = context.retriever.retrieve(query, top_k=settings.SEARCH_TOP_K,
                    user=context.user, version_manifest=context.version_manifest)
            items = []
            for doc in found:
                # 工具说明不能当作业务回答的事实依据。
                if doc.metadata.get("category") == "assistant_tools":
                    continue
                copied = Document(page_content=doc.page_content[:6000], metadata=dict(doc.metadata))
                copied.metadata["citation_id"] = len(context.documents) + 1
                context.documents.append(copied)
                items.append({"citation_id": copied.metadata["citation_id"], "content": copied.page_content,
                    "source": copied.metadata.get("original_filename") or copied.metadata.get("source", "未知来源"),
                    "page": copied.metadata.get("page"), "document_id": copied.metadata.get("document_id")})
            return {"status": "succeeded" if items else "no_results", "items": items,
                "message": "已返回可访问的资料" if items else "没有找到当前可访问的相关内部资料"}

        return [search_knowledge]


class CalculationTools:
    """引用已取得的真实结果进行计算。"""

    capability = "calculation"

    def __init__(self, context: AgentToolContext):
        self.context = context

    def get_tools(self) -> list[BaseTool]:
        context = self.context

        @tool("calculate", args_schema=CalculationArgs)
        def calculate_from_evidence(operation: str, references: list[CellRef]) -> dict:
            """引用查询结果行列计算；difference/ratio/percent_change依次填写本期、基准。"""
            value = calculate(operation, references, context.evidence)
            key = "calc_" + uuid4().hex[:12]
            value["evidence_id"] = key
            context.evidence[key] = value
            context.current_ids.add(key)
            return value

        return [calculate_from_evidence]


class ConversationTools:
    """按需开放工具，以及保存最终回答和经验证的任务信息。"""

    def __init__(self, context: AgentToolContext, capability_tools: dict[str, set[str]]):
        self.context = context
        self.capability_tools = capability_tools

    def get_tools(self) -> list[BaseTool]:
        context = self.context

        @tool(args_schema=DiscoverArgs)
        def discover_capabilities(capabilities: list[str]) -> dict:
            """按需获取business/knowledge/calculation相关能力的完整工具说明。无需工具时直接完成回答。"""
            if set(capabilities) - context.capabilities.keys():
                raise ValueError("请求的能力当前不可用")
            for capability in capabilities:
                context.available.update(self.capability_tools[capability])
            value = {"status": "succeeded", "tools": [
                {"name": name, "description": context.registry[name].description}
                for name in sorted(context.available)]}
            if "business" in capabilities:
                value.update(tables=schema_catalog(context.user), rules=BUSINESS_RULES)
            return value

        @tool(args_schema=FinishArgs)
        def finish_answer(answer: str, response_type: str = "answer", goal: str = "",
                facts: dict[str, CellRef] | None = None, reset_task: bool = False,
                pending_questions: list[str] | None = None) -> dict:
            """返回最终答案或追问，并保存目标、经结果验证的对象/场次/时间信息。查过业务数据后请使用此工具完成。"""
            args = FinishArgs(answer=answer, response_type=response_type, goal=goal, facts=facts or {},
                reset_task=reset_task, pending_questions=pending_questions or [])
            # 保存的对象ID等信息必须引用真实结果，不能由模型凭空填写。
            verified_facts = {k: {**r.model_dump(), "value": cell_value(context.evidence, r)}
                for k, r in args.facts.items()}
            state = context.task_state
            changed = any(k in state.get("facts", {}) and state["facts"][k].get("value") != v["value"]
                for k, v in verified_facts.items())
            if args.reset_task or changed:
                keep = context.current_ids | {v["evidence_id"] for v in verified_facts.values()}
                kept = {k: v for k, v in context.evidence.items() if k in keep}
                context.evidence.clear()
                context.evidence.update(kept)
                state.clear()
                state.update(facts={}, evidence=context.evidence)
            state.setdefault("facts", {}).update(verified_facts)
            state["goal"] = args.goal or state.get("goal") or context.question
            state["pending_questions"] = args.pending_questions
            context.final = args
            return {"status": "succeeded", "message": "已形成回答并整理任务信息"}

        return [discover_capabilities, finish_answer]


def build_agent_tools(context: AgentToolContext) -> dict[str, BaseTool]:
    """从四类工具的真实定义生成注册表，说明与执行共用同一份定义。"""
    registry, capability_tools = {}, {}
    for group in (BusinessTools(context), KnowledgeTools(context), CalculationTools(context)):
        tools = group.get_tools()
        registry.update((item.name, item) for item in tools)
        capability_tools[group.capability] = {item.name for item in tools}
    registry.update((item.name, item) for item in ConversationTools(context, capability_tools).get_tools())
    context.registry = registry
    return registry
