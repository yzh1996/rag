"""Dictionary-backed business schema. MySQL storage mappings are explicit."""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import re

from sqlalchemy import (BigInteger, Boolean, Column, Date, DateTime, Index,
                        Integer, JSON, MetaData, Numeric, SmallInteger, String, Table, Text)
from sqlalchemy.dialects.mysql import LONGTEXT, MEDIUMTEXT

CATALOG_PATH = Path(__file__).resolve().parents[1] / "config" / "business_schema.json"
SOURCE_TABLES = {"shop", "jl_user", "live_base", "live_info_traffic", "live_product",
                 "live_product_sale", "live_promotion_product_id", "live_sales_info",
                 "product", "product_brand", "anchor_video_item", "anchor_video_item_comment",
                 "date_product_video_incr", "date_video_data"}
# Constraints for the synthetic dataset, not claims about undocumented production keys.
DATASET_KEYS = {"shop": ["shop_id"], "jl_user": ["anchor_id"], "live_base": ["id"],
                "product": ["id"], "product_brand": ["id"], "anchor_video_item": ["id"],
                "anchor_video_item_comment": ["id"], "live_info_traffic": ["_id"],
                "live_sales_info": ["room_id", "date_flag"],
                "live_product_sale": ["room_id", "product_id", "date_flag"],
                "live_product": ["room_id", "promotion_id", "date_flag"],
                "live_promotion_product_id": ["promotion_id", "product_id"],
                "date_video_data": ["date_flag", "video_id", "product_id"],
                "date_product_video_incr": ["date_flag", "video_id", "product_id"]}


@lru_cache(maxsize=1)
def load_business_catalog():
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def catalog_hash():
    return hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest()


def storage_type(source_type):
    """Keep unknown definitions as nullable TEXT; never infer business meaning."""
    value = source_type.lower().strip()
    if value.startswith("array<"):
        return JSON()
    if value.startswith("bigint"):
        return BigInteger()
    if value.startswith(("tinyint", "smallint")):
        return SmallInteger()
    if value.startswith("int"):
        return Integer()
    if value == "boolean":
        return Boolean()
    if value == "datetime":
        return DateTime()
    if value == "date":
        return Date()
    match = re.fullmatch(r"decimal\((\d+),(\d+)\)", value)
    if match:
        return Numeric(int(match[1]), int(match[2]))
    match = re.fullmatch(r"varchar\((\d+)\)", value)
    if match:
        length = int(match[1])
        # The dictionary contains VARCHAR(65533); utf8mb4 cannot store that inline.
        if length > 4000:
            return Text().with_variant(MEDIUMTEXT(), "mysql")
        if length > 512:
            return Text()
        return String(length)
    if value == "源文档未提供":
        return Text()
    raise ValueError(f"未支持的字典类型：{source_type}")


def business_metadata():
    metadata = MetaData()
    for spec in load_business_catalog()["tables"]:
        if not spec["columns"]:
            continue
        keys = DATASET_KEYS.get(spec["name"], [])
        cols = []
        for col in spec["columns"]:
            known = col["source_type"] != "源文档未提供"
            cols.append(Column(col["name"], storage_type(col["source_type"]),
                               primary_key=col["name"] in keys,
                               nullable=False if col["name"] in keys else col["nullable"] != "NO" or not known,
                               comment=(col["description"] + "；字典类型：" + col["source_type"])[:900]))
        table = Table(spec["name"], metadata, *cols, comment="字段字典；本地数据为模拟数据",
                      mysql_charset="utf8mb4", mysql_collate="utf8mb4_bin")
        for field in ("anchor_id", "shop_id", "room_id", "date_flag", "live_create_time"):
            if field in table.c and field not in keys:
                Index(f"ix_biz_{table.name}_{field}", table.c[field])
    return metadata


def schema_status():
    tables = load_business_catalog()["tables"]
    return {"dictionary_tables": len(tables),
            "creatable_tables": sum(bool(t["columns"]) for t in tables),
            "pending_tables": [t["name"] for t in tables if not t["columns"]],
            "unresolved_fields": [{"table": t["name"], "column": c["name"]}
                                  for t in tables for c in t["columns"]
                                  if c["source_type"] == "源文档未提供" or c["description"] == "源文档未提供"]}


def query_schema(names=None):
    """Only documented fields are offered to generated SQL."""
    schemas = {t["name"]: {c["name"]: c["source_type"] for c in t["columns"]
                         if c["source_type"] != "源文档未提供" and c["description"] != "源文档未提供"}
            for t in load_business_catalog()["tables"]
            if t["columns"] and (names is None or t["name"] in names)}
    return {name: columns for name, columns in schemas.items() if columns}
