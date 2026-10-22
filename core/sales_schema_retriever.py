"""独立的可信表结构 RAG；索引仅保存结构和规则，不保存销量数据。"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re

from config.settings import settings

CATALOG_PATH = Path(__file__).resolve().parents[1] / "config" / "sales_schema.json"
COLLECTION_NAME = "sales_schema"
BUILD_HINT = "请运行 python scripts/build_sales_schema_index.py 重建表结构索引。"


class SalesSchemaError(ValueError):
    def __init__(self, message, code="schema_index_unavailable"):
        super().__init__(message)
        self.code = code
        self.repairable = False


def load_catalog(path=None):
    """只读取项目维护的目录，不接收上传文档作为表结构。"""
    try:
        catalog = json.loads(Path(path or CATALOG_PATH).read_text(encoding="utf-8"))
        tables = catalog["tables"]
        rules = {rule["id"]: rule for rule in catalog["rules"]}
        if len(rules) != len(catalog["rules"]) or not tables:
            raise ValueError("empty or repeated catalog entries")
        names = [table["name"] for table in tables]
        if len(set(names)) != len(names):
            raise ValueError("repeated tables")
        for table in tables:
            if (not re.fullmatch(r"[a-z][a-z0-9_]*", table["name"]) or not table["columns"] or
                    not table["aliases"] or not table["required_rules"] or
                    not set(table["primary_key"]) <= table["columns"].keys() or
                    not set(table["required_rules"]) <= rules.keys()):
                raise ValueError("invalid table definition")
            for name, column in table["columns"].items():
                if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or not column["type"] or not column["description"]:
                    raise ValueError("invalid column definition")
        payload = json.dumps(catalog, ensure_ascii=False, sort_keys=True)
        fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        documents = {}
        for table in tables:
            columns = ", ".join(f"{name} {column['type']}" for name, column in table["columns"].items())
            meanings = "\n".join(f"{name}：{column['description']}" for name, column in table["columns"].items())
            documents["table." + table["name"]] = {
                "content": f"{table['title']}\n{table['name']}({columns})\n{table['description']}\n"
                    f"记录粒度：{table['grain']}\n主键：{', '.join(table['primary_key'])}\n{meanings}",
                "kind": "table", "title": table["title"], "table": table["name"],
            }
        for rule_id, rule in rules.items():
            if rule_id in documents:
                raise ValueError("repeated document ids")
            documents[rule_id] = {"content": rule["title"] + "\n" + rule["content"],
                "kind": "rule", "title": rule["title"], "table": ""}
        return catalog, documents, fingerprint
    except (OSError, ValueError, KeyError, TypeError):
        raise SalesSchemaError("表结构与指标目录无效，请先检查 config/sales_schema.json。", "schema_catalog_invalid") from None


@lru_cache(maxsize=4)
def _client(directory):
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(path=directory, settings=Settings(anonymized_telemetry=False))


def _embedding_model():
    from core.embedding import get_embedding_model

    return get_embedding_model()


def build_sales_schema_index(*, index_directory=None, catalog_path=None, embedding_model=None):
    """人工/脚本更新独立集合；构建未完成的索引不能用于生成 SQL。"""
    _, documents, fingerprint = load_catalog(catalog_path)
    directory = Path(index_directory or settings.SALES_SCHEMA_INDEX_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    model = embedding_model or _embedding_model()
    ids = list(documents)
    contents = [documents[key]["content"] for key in ids]
    vectors = model.embed_documents(contents)
    client = _client(str(directory.resolve()))
    collection = client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}, embedding_function=None)
    existing = collection.get(limit=1, include=["embeddings"])
    if len(existing["ids"]) and len(existing["embeddings"][0]) != len(vectors[0]):
        # Chroma 不能原地改变向量维度；仅重建专用集合，不影响文档知识库。
        client.delete_collection(COLLECTION_NAME)
        collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"}, embedding_function=None)
    metadata = {"catalog_hash": fingerprint,
        "embedding_model": settings.EMBEDDING_MODEL_NAME, "status": "building"}
    collection.modify(metadata=metadata)
    collection.upsert(ids=ids, documents=contents, embeddings=vectors,
        metadatas=[{"kind": documents[key]["kind"], "table": documents[key]["table"],
            "catalog_hash": fingerprint} for key in ids])
    obsolete = sorted(set(collection.get(include=[])["ids"]) - set(ids))
    if obsolete:
        collection.delete(ids=obsolete)
    collection.modify(metadata=dict(metadata, status="ready"))
    return {"documents": len(ids), "tables": sum(doc["kind"] == "table" for doc in documents.values()),
        "catalog_hash": fingerprint, "index_directory": str(directory.resolve())}


@dataclass
class SalesSchemaSelection:
    tables: list
    context: str
    trace: dict

    def verify(self, connection, allowed_schema):
        """仅用受控 PRAGMA 核对结构、类型和主键，不读取业务行。"""
        for table in self.tables:
            name = table["name"]
            if name not in allowed_schema or set(table["columns"]) != set(allowed_schema[name]):
                raise SalesSchemaError("检索到的表或字段不在当前销量查询白名单中。", "schema_not_allowed")
            rows = connection.execute(f'PRAGMA table_info("{name}")').fetchall()
            actual = {row["name"]: re.sub(r"\s+", "", row["type"].upper()) for row in rows}
            expected = {key: re.sub(r"\s+", "", col["type"].upper()) for key, col in table["columns"].items()}
            primary_key = [row["name"] for row in sorted(rows, key=lambda row: row["pk"]) if row["pk"]]
            if actual != expected or primary_key != table["primary_key"]:
                raise SalesSchemaError("表结构目录与当前数据库不一致，请核对表字段、类型和主键后重建索引。", "schema_changed")
        self.trace["status"] = "validated"

    @property
    def table_names(self):
        return [table["name"] for table in self.tables]


class SalesSchemaRetriever:
    def __init__(self, *, index_directory=None, catalog_path=None, embedding_model=None):
        self.index_directory = Path(index_directory or settings.SALES_SCHEMA_INDEX_DIR)
        self.catalog_path = catalog_path
        self.embedding_model = embedding_model

    @staticmethod
    def _previous_scope(history, tables):
        # 只沿用最近一轮查数；不把结果中的销量行送去向量化。
        for message in reversed(history):
            if message.type == "ai":
                details = message.additional_kwargs.get("query_details") or {}
                names = details.get("schema_retrieval", {}).get("tables", [])
                if not names and details.get("sql"):
                    # 兼容升级前保存的查数历史，只提取已执行 SQL 的表名，不复制旧结构。
                    from sqlglot import exp, parse_one
                    from sqlglot.errors import ParseError
                    try:
                        names = [table.name.lower() for table in parse_one(details["sql"], read="sqlite").find_all(exp.Table)]
                    except ParseError:
                        names = []
                return [table["name"] for table in tables if table["name"] in names]
        return []

    def retrieve(self, question, history):
        catalog, documents, fingerprint = load_catalog(self.catalog_path)
        if not (self.index_directory / "chroma.sqlite3").is_file():
            raise SalesSchemaError("尚未建立独立的表结构索引。" + BUILD_HINT)
        try:
            collection = _client(str(self.index_directory.resolve())).get_collection(COLLECTION_NAME, embedding_function=None)
            metadata = collection.metadata or {}
            if metadata.get("catalog_hash") != fingerprint or metadata.get("embedding_model") != settings.EMBEDDING_MODEL_NAME:
                raise SalesSchemaError("表结构目录或向量模型已变化。" + BUILD_HINT, "schema_index_stale")
            if metadata.get("status") != "ready" or collection.count() != len(documents):
                raise SalesSchemaError("表结构索引未构建完成。" + BUILD_HINT, "schema_index_incomplete")
            explicit = [table["name"] for table in catalog["tables"] if any(
                alias.lower() in question.lower() for alias in table["aliases"])]
            last_scope = self._previous_scope(history, catalog["tables"])
            cross_channel = bool(re.search(r"两(?:个|种)?渠道|双渠道|全渠道|所有渠道|跨渠道|两者", question))
            previous = last_scope if not explicit else []
            if cross_channel:
                explicit = [table["name"] for table in catalog["tables"]]
                previous = []
            elif explicit and re.search(r"加上|算上|再加|一并|一起|也算", question):
                explicit = [table["name"] for table in catalog["tables"] if table["name"] in {*explicit, *last_scope}]
            query = question
            if previous:
                titles = [table["title"] for table in catalog["tables"] if table["name"] in previous]
                query += "\n上一轮查询范围：" + "、".join(titles)
            model = self.embedding_model or _embedding_model()
            result = collection.query(query_embeddings=[model.embed_query(query)], n_results=min(6, len(documents)),
                include=["documents", "metadatas", "distances"])
            hits = {}
            for key, content, attributes, distance in zip(result["ids"][0], result["documents"][0],
                    result["metadatas"][0], result["distances"][0]):
                if key not in documents or content != documents[key]["content"] or (attributes or {}).get("catalog_hash") != fingerprint:
                    raise SalesSchemaError("表结构索引内容与可信目录不一致。" + BUILD_HINT, "schema_index_stale")
                similarity = 1 - float(distance)
                if not math.isfinite(similarity):
                    raise SalesSchemaError("表结构检索未返回有效的匹配分数。", "schema_not_found")
                hits[key] = {"id": key, "title": documents[key]["title"],
                    "similarity": round(similarity, 4), "kind": documents[key]["kind"]}
            required = explicit or previous
            if not required and re.search(r"销量|销售量|主播|带货|成交件数|卖", question):
                # 未限定渠道的总销量/主播查数需要两渠道，不能因为 top1 漏掉另一张表。
                required = [table["name"] for table in catalog["tables"]]
            if not required:
                required = [table["name"] for table in catalog["tables"]
                    if hits.get("table." + table["name"], {}).get("similarity", -1) >= settings.SALES_SCHEMA_MIN_SIMILARITY]
            selected = []
            for name in required:
                hit = hits.get("table." + name)
                # 关键词本身也是召回证据；无关键词或历史依据时必须满足向量相关度。
                if not hit or (name not in explicit and name not in previous and hit["similarity"] < settings.SALES_SCHEMA_MIN_SIMILARITY):
                    raise SalesSchemaError("未完整检索到问题需要的表结构，请明确直播/视频查询范围，或检查表结构索引。", "schema_not_found")
                hit["reason"] = "本轮渠道关键词" if name in explicit else "上一轮查数范围" if name in previous else "向量匹配与渠道覆盖检查"
                selected.append(next(table for table in catalog["tables"] if table["name"] == name))
            if not selected:
                raise SalesSchemaError("未检索到相关表结构，请明确要查询的销量数据。", "schema_not_found")
            rule_ids = list(dict.fromkeys(rule for table in selected for rule in table["required_rules"]))
            # 规则是表的必需依赖，不能因为向量排名低就省略。
            rules = collection.get(ids=rule_ids, include=["documents", "metadatas"])
            found_rules = dict(zip(rules["ids"], zip(rules["documents"], rules["metadatas"])))
            for rule_id in rule_ids:
                content, attributes = found_rules.get(rule_id, (None, {}))
                if content != documents[rule_id]["content"] or (attributes or {}).get("catalog_hash") != fingerprint:
                    raise SalesSchemaError("查询需要的指标规则不完整。" + BUILD_HINT, "schema_index_incomplete")
            selected_ids = ["table." + table["name"] for table in selected] + rule_ids
            trace = {"status": "selected", "query": query, "tables": [table["name"] for table in selected],
                "rules": rule_ids, "catalog_hash": fingerprint,
                "hits": [dict(hits.get(key, {"id": key, "title": documents[key]["title"], "kind": "rule"}),
                    **({"reason": "所选表必需的指标规则"} if key in rule_ids else {})) for key in selected_ids]}
            return SalesSchemaSelection(selected, "\n\n".join(documents[key]["content"] for key in selected_ids), trace)
        except SalesSchemaError:
            raise
        except Exception:
            raise SalesSchemaError("表结构检索暂时不可用，请检查独立索引与向量模型。" + BUILD_HINT) from None


def get_sales_schema_retriever():
    return SalesSchemaRetriever()
