"""Current business source: MySQL dictionary tables; legacy fixtures remain isolated."""
from statistics import median

from core import commerce_demo_data as legacy
from core.business_reader import (anchor_mentions, analyze_business_anchor, find_business_anchors,
                                  mysql_business_enabled)

calculate_channel_metrics = legacy.calculate_channel_metrics


def find_anchors(name):
    return find_business_anchors(name) if mysql_business_enabled() else legacy.find_anchors(name)


def find_anchor_mentions(value):
    return anchor_mentions(value) if mysql_business_enabled() else legacy.find_anchor_mentions(value)


def analyze_anchor_live(anchor_id, start_date, end_date):
    if mysql_business_enabled():
        return analyze_business_anchor(anchor_id, start_date, end_date, "live")
    return legacy.analyze_anchor_live(anchor_id, start_date, end_date)


def analyze_anchor_video(anchor_id, start_date, end_date):
    if mysql_business_enabled():
        return analyze_business_anchor(anchor_id, start_date, end_date, "video")
    return legacy.analyze_anchor_video(anchor_id, start_date, end_date)


def analyze_anchor_products(anchor_id, start_date, end_date):
    if not mysql_business_enabled():
        return legacy.analyze_anchor_products(anchor_id, start_date, end_date)
    live = analyze_anchor_live(anchor_id, start_date, end_date)
    video = analyze_anchor_video(anchor_id, start_date, end_date)
    products = {}
    for row in live["products"] + video["products"]:
        item = products.setdefault(row["product_id"], {**row, "sales_quantity": 0, "gmv": 0})
        item["sales_quantity"] += row["sales_quantity"]
        item["gmv"] += row["gmv"]
    ordered = sorted(products.values(), key=lambda p: (-p["sales_quantity"], p["product_id"]))
    total = sum(p["sales_quantity"] for p in ordered)
    return {"anchor_id": anchor_id, "anchor_name": live["anchor_name"], "period_start": start_date,
            "period_end": end_date, "total_sales_quantity": total, "products": ordered,
            "top3_product_share": round(sum(p["sales_quantity"] for p in ordered[:3]) / total, 4) if total else 0,
            "categories": [{"category": live["main_category"], "sales_quantity": total,
                            "gmv": live["gmv"] + video["gmv"], "sales_share": 1 if total else 0}],
            "dataset_label": "SIMULATED"}


def compare_anchor_category(anchor_id, start_date, end_date):
    if not mysql_business_enabled():
        return legacy.compare_anchor_category(anchor_id, start_date, end_date)
    current = find_anchors(str(anchor_id))[0]
    peers = [a for a in find_business_anchors() if a["anchor_id"] != current["anchor_id"]
             and a["main_category"] == current["main_category"]]
    def quantity(aid):
        return sum(analyze_business_anchor(aid, start_date, end_date, channel)["sales_quantity"]
                   for channel in ("live", "video"))
    values = [{**a, "total_sales_quantity": quantity(a["anchor_id"])} for a in peers]
    return {"anchor_id": anchor_id, "own_total_sales_quantity": quantity(anchor_id), "peer_count": len(values),
            "peers": values, "peer_median_sales_quantity": median(p["total_sales_quantity"] for p in values) if values else None,
            "sample_size_warning": "同类模拟账号样本较少，不能代表行业整体。", "dataset_label": "SIMULATED"}
