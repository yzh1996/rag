"""Bounded read-only access to dictionary business data, shared by analysis and SQL."""
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from decimal import Decimal
import json
import re
import threading

from sqlalchemy import select, text

from config.settings import settings
from core import database
from core.business_seed import dataset_table

_SLOTS = threading.BoundedSemaphore(4)


def mysql_business_enabled():
    return database.engine.dialect.name == "mysql"


def scalar_value(value):
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat(sep=" ") if isinstance(value, datetime) else value.isoformat()
    return value


@contextmanager
def readonly_business_connection(engine=None, cancelled=None):
    engine = engine or database.engine
    if not _SLOTS.acquire(blocking=False):
        raise ValueError("当前业务查询较多，请稍后重试")
    try:
        with engine.connect() as connection:
            if cancelled is not None and cancelled.is_set():
                raise ValueError("查询已取消")
            if engine.dialect.name == "mysql":
                timeout = max(10, min(int(settings.SALES_QUERY_TIMEOUT_SECONDS * 1000), 30000))
                connection.exec_driver_sql(f"SET SESSION MAX_EXECUTION_TIME = {timeout}")
                connection.exec_driver_sql("SET TRANSACTION READ ONLY")
                connection.exec_driver_sql("START TRANSACTION WITH CONSISTENT SNAPSHOT")
            try:
                yield connection
            finally:
                connection.rollback()
    finally:
        _SLOTS.release()


def rows(connection, sql, params=None):
    return [{key: scalar_value(value) for key, value in row.items()}
            for row in connection.execute(text(sql), params or {}).mappings()]


def dataset_info(connection):
    row = connection.execute(select(dataset_table())).mappings().first()
    if not row:
        raise ValueError("业务数据尚未初始化，请先运行 scripts/seed_mysql_business.py")
    return {key: scalar_value(value) for key, value in row.items()}


def find_business_anchors(name=None, engine=None):
    with readonly_business_connection(engine) as connection:
        items = rows(connection, """SELECT anchor_id, user_name AS anchor_name,
            first_tag_name AS main_category, follow_count AS follower_count, shop_id
            FROM jl_user WHERE COALESCE(is_delete,0)=0 ORDER BY anchor_id LIMIT 200""")
    for item in items:
        item["anchor_id"] = str(item["anchor_id"])
    if name is None:
        return items
    normalized = str(name).strip().casefold()
    exact = [item for item in items if str(item["anchor_id"]) == normalized or item["anchor_name"].casefold() == normalized]
    return exact or [item for item in items if normalized and normalized in item["anchor_name"].casefold()]


def anchor_mentions(value, engine=None):
    return [item for item in find_business_anchors(engine=engine)
            if item["anchor_name"].casefold() in value.casefold() or
            re.search(r"(?<!\d)" + re.escape(str(item["anchor_id"])) + r"(?!\d)", value)]


def _validate_window(start, end):
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last or (last - first).days >= settings.COMMERCE_DEMO_MAX_PERIOD_DAYS:
        raise ValueError("查询日期应按先后排列，且不超过允许的分析周期")


def analyze_business_anchor(anchor_id, start_date, end_date, channel="live", engine=None):
    _validate_window(start_date, end_date)
    matches = find_business_anchors(str(anchor_id), engine)
    if len(matches) != 1:
        raise ValueError("请先确认唯一的主播ID")
    anchor = matches[0]
    params = {"aid": int(anchor_id), "start": start_date, "end": end_date,
              "end_exclusive": (date.fromisoformat(end_date) + timedelta(days=1)).isoformat()}
    with readonly_business_connection(engine) as connection:
        info = dataset_info(connection)
        if channel == "live":
            summary = rows(connection, """SELECT COUNT(*) AS live_count,COUNT(DISTINCT s.date_flag) AS active_days,
                COALESCE(SUM(s.sales),0) AS gmv,COALESCE(SUM(l.total_user),0) AS viewers
                FROM live_sales_info s JOIN live_base l ON l.id=s.room_id WHERE s.anchor_id=:aid
                AND s.date_flag BETWEEN :start AND :end AND l.finish_time>0 AND COALESCE(l.is_show,0)=0""", params)[0]
            product_rows = rows(connection, """SELECT x.product_id,p.title AS product_name,p.first_cid,
                SUM(x.number) AS sales_quantity,SUM(x.number*x.min_price) AS gmv
                FROM live_product_sale x JOIN product p ON p.id=CAST(x.product_id AS CHAR)
                WHERE x.anchor_id=:aid AND x.date_flag BETWEEN :start AND :end
                GROUP BY x.product_id,p.title,p.first_cid ORDER BY sales_quantity DESC,x.product_id LIMIT 100""", params)
        else:
            summary = rows(connection, """SELECT COUNT(DISTINCT video_id) AS video_count,COALESCE(SUM(sales),0) AS gmv
                FROM date_product_video_incr WHERE anchor_id=:aid AND date_flag BETWEEN :startday AND :endday""",
                {**params, "startday": int(start_date.replace("-", "")), "endday": int(end_date.replace("-", ""))})[0]
            product_rows = rows(connection, """SELECT x.product_id,p.title AS product_name,p.first_cid,
                SUM(x.sales_volume) AS sales_quantity,SUM(x.sales) AS gmv FROM date_product_video_incr x
                JOIN product p ON p.id=CAST(x.product_id AS CHAR) WHERE x.anchor_id=:aid
                AND x.date_flag BETWEEN :startday AND :endday GROUP BY x.product_id,p.title,p.first_cid
                ORDER BY sales_quantity DESC,x.product_id LIMIT 100""",
                {**params, "startday": int(start_date.replace("-", "")), "endday": int(end_date.replace("-", ""))})
            summary["plays"] = None
    total = sum(r["sales_quantity"] for r in product_rows)
    return {**anchor, **summary, "sales_quantity": total, "period_start": start_date, "period_end": end_date,
            "categories": [{"category": anchor["main_category"], "sales_quantity": total, "gmv": summary["gmv"]}],
            "top_products": product_rows[:5], "products": product_rows, "dataset_label": info["dataset_label"]}
