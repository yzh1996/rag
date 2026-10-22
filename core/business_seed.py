"""One coherent synthetic dataset using dictionary table and field names."""
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
import json

from sqlalchemy import Column, Date, Integer, JSON, MetaData, String, Table, func, inspect, select

from core.business_schema import business_metadata, catalog_hash, load_business_catalog, schema_status

SEED_VERSION = "dictionary-v1"
ASSUMPTIONS = [
    "所有经营记录均为本地模拟数据，不代表真实店铺或主播。",
    "模拟约定：jl_user.shop_id 表示自播店铺，live_base.id 对应 room_id；不代表已确认真实库的关联规则。",
    "模拟约定：live_product_sale.number 为成交件数；live_sales_info.sales 为人民币元，整数金额。",
    "模拟商品采用整数元固定成交价；商品金额为 number × min_price 的估算，直播销售额以 live_sales_info 为准。",
    "BIGINT 日期标记使用 YYYYMMDD；INT 时间使用 Unix 秒，DATETIME 使用北京时间。",
    "数组在 MySQL 存为 JSON，超长 VARCHAR 存为 TEXT；未定义类型存为可空 TEXT，未确认字段保留 NULL。",
    "仅有表名没有字段的表不建表；不把占位文字当字段，也不猜测缺失结构。",
    "明细、汇总和排行来自同一批基础记录；禁止把不同加工层重复相加。",
]


def dataset_table(metadata=None):
    return Table("business_datasets", metadata or MetaData(),
                 Column("id", Integer, primary_key=True), Column("seed_version", String(64), nullable=False),
                 Column("catalog_hash", String(64), nullable=False), Column("dataset_label", String(32), nullable=False),
                 Column("end_date", Date, nullable=False), Column("counts", JSON, nullable=False),
                 Column("assumptions", JSON, nullable=False))


def _epoch(value):
    return int(value.replace(tzinfo=timezone(timedelta(hours=8))).timestamp())


def build_records(end_date=None, days=90):
    end_date = end_date or date.today() - timedelta(days=1)
    if not 1 <= days <= 120:
        raise ValueError("模拟周期必须在1至120天之间")
    records = {t["name"]: [] for t in load_business_catalog()["tables"] if t["columns"]}
    start = end_date - timedelta(days=days - 1)
    names = ["陈洁KiKi", "美妆阿沐", "日常好物安安", "小美吃货", "穿搭小青", "生活研究所"]
    shops = ["清研美妆旗舰店", "日光美妆店", "简居生活旗舰店", "山野食品旗舰店", "云衣服饰旗舰店", "简居家居店"]
    categories = ["美妆个护", "美妆个护", "家居日用", "食品饮料", "服饰穿搭", "家居日用"]
    products = [("温和洁面套装", 129), ("补水面膜组合", 99), ("防晒乳双支装", 159), ("头皮护理套装", 179),
                ("家用收纳套组", 89), ("轻音加湿器", 229), ("坚果分享礼盒", 118), ("低糖燕麦组合", 79),
                ("轻薄防晒外套", 189), ("通勤基础针织衫", 219), ("厨房清洁组合", 69), ("桌面氛围灯", 139)]
    now = datetime.combine(end_date, time(23, 0))
    for i, name in enumerate(names):
        aid, sid = 1001 + i, 101 + i
        records["shop"].append({"shop_id": sid, "name": shops[i], "create_time": datetime.combine(start, time()),
                                "shop_author_id": aid, "main_cate": [categories[i]], "main_cate_text": categories[i],
                                "product_num": 2, "score": Decimal("4.80"), "update_time": now})
        records["jl_user"].append({"anchor_id": aid, "user_name": name, "shop_id": sid,
                                   "first_tag_name": categories[i], "follow_count": 90000 + i * 23000,
                                   "follow_incr": 200 + i * 30, "city": "杭州", "province": "浙江",
                                   "is_delete": False, "update_time": _epoch(now)})
        records["product_brand"].append({"id": str(301 + i), "brand_id": 301 + i, "name": shops[i].replace("旗舰店", ""),
                                         "create_time": datetime.combine(start, time()), "update_time": now})
        for j in range(2):
            pid = 2001 + i * 2 + j
            title, price = products[i * 2 + j]
            records["product"].append({"id": str(pid), "shop_id": sid, "promotion_id": str(5000 + pid),
                "title": title, "brand_name": records["product_brand"][-1]["name"], "min_price": Decimal(price),
                "max_price": Decimal(price), "cos_ratio": Decimal("10.00"), "first_cid": i + 1,
                "status": 1, "sales": 0, "create_time": datetime.combine(start, time()), "update_time": now})
            records["live_promotion_product_id"].append({"promotion_id": 5000 + pid, "product_id": pid})
        for offset in range(days):
            day = start + timedelta(days=offset)
            # Multiple sessions on some dates make "latest session" different from "latest day".
            sessions = 2 if offset % 7 == 0 or day == end_date else 1
            for session in range(sessions):
                room = 1000000 + i * 10000 + offset * 10 + session
                begun = datetime.combine(day, time(10 if session == 0 else 19))
                viewers = 3000 + offset * 31 + i * 400 + session * 200
                live = {"id": room, "anchor_id": aid, "live_create_time": begun,
                        "finish_time": _epoch(begun + timedelta(hours=2)), "is_show": False,
                        "title": shops[i] + "日常带货", "total_user": viewers, "top_user": viewers // 3,
                        "exposed_num": viewers * 4, "follow_count": 25 + i, "like_count": viewers * 2,
                        "update_time": _epoch(begun + timedelta(hours=2))}
                records["live_base"].append(live)
                amount = 0
                for j in range(2):
                    pid = 2001 + i * 2 + j
                    quantity = 20 + (offset * 7 + i * 13 + j * 11 + session * 17) % 90
                    price = products[i * 2 + j][1]
                    amount += quantity * price
                    records["product"][i * 2 + j]["sales"] += quantity
                    row = {"room_id": room, "product_id": pid, "date_flag": day, "anchor_id": aid,
                           "number": quantity, "min_price": Decimal(price)}
                    records["live_product_sale"].append(row)
                    records["live_product"].append({"room_id": room, "promotion_id": 5000 + pid,
                                                   "date_flag": day, "anchor_id": aid, "number": quantity})
                records["live_sales_info"].append({"room_id": room, "date_flag": day, "anchor_id": aid,
                    "sales_low": amount, "sales_high": amount, "sales": amount, "product_num": 2})
                records["live_info_traffic"].append({"_id": room, "anchor_id": aid, "date_flag": day,
                    "city_wide": Decimal("5"), "follow": Decimal("15"), "home_page": Decimal("5"),
                    "live_plaza": Decimal("10"), "other": Decimal("5"), "recommend_freed": Decimal("45"),
                    "search": Decimal("5"), "short_video_diversion": Decimal("10")})
            vid = 2000000 + i * 10000 + offset
            pid = 2001 + i * 2 + offset % 2
            quantity = 5 + (offset * 3 + i * 7) % 30
            price = products[pid - 2001][1]
            published = datetime.combine(day, time(9))
            records["product"][pid - 2001]["sales"] += quantity
            records["anchor_video_item"].append({"id": vid, "anchor_id": aid, "publish_date_time": published,
                "duration": 45, "like_cnt": quantity * 20, "coment_cnt": quantity * 2, "share_cnt": quantity,
                "video_text": shops[i] + "商品介绍", "product_id": str(pid), "product_ids_array": [str(pid)],
                "cart_video": True, "is_graphic": False, "is_delete": False, "update_time": _epoch(published)})
            records["date_product_video_incr"].append({"date_flag": int(day.strftime("%Y%m%d")), "video_id": vid,
                "product_id": pid, "publish_date": _epoch(published), "anchor_id": aid, "product_count": 1,
                "sales": quantity * price, "sales_volume": quantity, "create_time": _epoch(published)})
            records["date_video_data"].append({"date_flag": day, "video_id": str(vid), "product_id": str(pid),
                "anchor_id": aid, "publish_date": day, "sales_low": quantity * price,
                "sales_high": quantity * price, "digg": quantity * 20, "comment": quantity * 2,
                "share": quantity, "collect": quantity, "is_delete": 0, "create_time": published})
            records["anchor_video_item_comment"].append({"id": vid + 9000000, "comment_time": published,
                "item_id": vid, "text": "模拟评论：希望了解商品规格", "nickname": "模拟用户",
                "user_id": 8000 + i, "like_count": 3, "reply_comment_total": 0, "update_time": _epoch(published)})
        # An unfinished stream must never win "latest completed live".
        begun = datetime.combine(end_date + timedelta(days=1), time(10))
        records["live_base"].append({"id": 9000000 + i, "anchor_id": aid, "live_create_time": begun,
                                     "finish_time": 0, "is_show": True, "title": "模拟进行中直播"})
    _derive_records(records, end_date)
    return records


def _derive_records(data, end):
    anchors = {a["anchor_id"]: a for a in data["jl_user"]}
    live = {r["id"]: r for r in data["live_base"] if r.get("finish_time")}
    live_info = {r["room_id"]: r for r in data["live_sales_info"]}
    products = {int(p["id"]): p for p in data["product"]}
    for target, source in {"dwd_live_product": "live_product", "dwd_live_product_sale": "live_product_sale",
                           "dwd_live_promotion_product_id": "live_promotion_product_id",
                           "dwd_live_sales_info": "live_sales_info"}.items():
        data[target] = list(data[source])
    data["ods_live_info_traffic"] = [{**r, "room_id": r["_id"]} for r in data["live_info_traffic"]]
    data["ods_jl_user_fans_portrait"] = [{"anchor_id": a["anchor_id"], "city": a["city"], "province": a["province"]}
                                         for a in anchors.values()]
    data["dwd_jl_user_detail"] = [{**a, "follow_cnt": a["follow_count"], "date_flag": end} for a in anchors.values()]
    data["dws_anchor_summary_detail"] = [{**a, "follow_cnt": a["follow_count"]} for a in anchors.values()]
    data["dws_live_product_info"] = [{**r, "promotion_id": 5000 + r["product_id"],
        "sales_low": int(r["min_price"]), "sales_high": int(r["min_price"]), "sales": int(r["min_price"]),
        "product_num": r["number"]} for r in data["live_product_sale"]]
    data["dws_live_summary_detail"] = [{**anchors[r["anchor_id"]], **r, "room_id": r["id"],
        "follow_cnt": anchors[r["anchor_id"]]["follow_count"], "like_cnt": r["like_count"]} for r in live.values()]
    data["dws_video_summary_detail"] = [{**anchors[v["anchor_id"]], **v, "video_id": v["id"],
        "follow_cnt": anchors[v["anchor_id"]]["follow_count"], "date_flag": v["publish_date_time"].date()}
        for v in data["anchor_video_item"]]
    by_room = defaultdict(int)
    for r in data["live_product_sale"]:
        by_room[r["room_id"]] += r["number"]
    # ads_live_product_sale_info.sales is NOT NULL but has no documented meaning.
    # Keep that table empty instead of silently treating it as the similarly named source metric.
    for name in ["ads_anchor_fans_rank1d", "ads_anchor_fans_rank7d", "ads_anchor_fans_rank30d"]:
        data[name] = [{**a, "follow_cnt": a["follow_count"], "product_num": 2} for a in anchors.values()]
    for days in (1, 7, 30):
        start = end - timedelta(days=days - 1)
        rows = []
        for aid, a in anchors.items():
            amount = sum(r["sales"] for r in data["live_sales_info"] if r["anchor_id"] == aid and r["date_flag"] >= start)
            amount += sum(r["sales"] for r in data["date_product_video_incr"] if r["anchor_id"] == aid and r["date_flag"] >= int(start.strftime("%Y%m%d")))
            rows.append({**a, "follow_cnt": a["follow_count"], "sales": amount})
        data[f"ads_anchor_sale_rank{days}d"] = sorted(rows, key=lambda r: (-r["sales"], r["anchor_id"]))
    for p in products.values():
        pid = int(p["id"])
        base = {**p, "product_id": pid}
        data["dws_product_summary_detail"].append(base)
        for name in ("ads_product_90_rank", "ads_product_his_rank", "ads_product_live_rank",
                     "ads_product_potential_rank", "ads_product_push_rank", "ads_product_sales_rank", "ads_product_video_rank"):
            if name == "ads_product_video_rank":
                amount = sum(r["sales"] for r in data["date_product_video_incr"] if r["product_id"] == pid)
            else:
                amount = sum(r["number"] * int(r["min_price"]) for r in data["live_product_sale"] if r["product_id"] == pid)
                if name in {"ads_product_sales_rank", "ads_product_push_rank"}:
                    amount += sum(r["sales"] for r in data["date_product_video_incr"] if r["product_id"] == pid)
            data[name].append({**base, "sales": amount})
    for shop in data["shop"]:
        aids = {aid for aid, a in anchors.items() if a["shop_id"] == shop["shop_id"]}
        amount = sum(r["sales"] for r in data["live_sales_info"] if r["anchor_id"] in aids)
        video_amount = sum(r["sales"] for r in data["date_product_video_incr"] if r["anchor_id"] in aids)
        data["dws_shop_summary_detail"].append({**shop, "sales_low": amount, "sales_high": amount})
        data["ads_shop_sale_rank"].append({**shop, "sales": amount + video_amount})
        brand = data["product_brand"][shop["shop_id"] - 101]
        data["dws_brand_summary_detail"].append({**brand, "sales": amount + video_amount, "product_num": 2})
    for video in data["anchor_video_item"]:
        sale = next(r for r in data["date_product_video_incr"] if r["video_id"] == video["id"])
        p = products[int(video["product_id"])]
        row = {**anchors[video["anchor_id"]], **video, "video_id": video["id"], "product_id": int(p["id"]),
               "follow_cnt": anchors[video["anchor_id"]]["follow_count"], "sales": sale["sales"],
               "date_flag": video["publish_date_time"].date(), "cos_ratio": p["cos_ratio"]}
        data["ads_video_hot_rank"].append(row)
        data["ads_video_sale_rank"].append(row)
    for a in anchors.values():
        relation = {"shop_id": a["shop_id"], "anchor_id": a["anchor_id"], "date_flag": end}
        data["dim_shop_anchor_rela_detail"].append(relation)
        data["dim_brand_anchor_rela_detail"].append(relation)
        for p in products.values():
            if p["shop_id"] == a["shop_id"]:
                data["dim_product_anchor_detail"].append({**p, **relation, "product_id": int(p["id"])})
    # Only fields with documented definitions are populated in derived tables.
    specs = {t["name"]: t for t in load_business_catalog()["tables"]}
    for name in data:
        allowed = {c["name"] for c in specs[name]["columns"]
                   if c["source_type"] != "源文档未提供" and
                   (name in {"live_info_traffic"} or c["description"] != "源文档未提供")}
        data[name] = [{key: value for key, value in row.items() if key in allowed} for row in data[name]]


def seed_business_data(engine, end_date=None, days=90):
    """Explicit only: refuse to seed over any pre-existing business records."""
    metadata = business_metadata()
    manifest = dataset_table()
    existing = set(inspect(engine).get_table_names())
    with engine.connect() as connection:
        if manifest.name in existing:
            previous = connection.execute(select(manifest)).mappings().first()
            if previous:
                if previous["catalog_hash"] != catalog_hash() or previous["seed_version"] != SEED_VERSION:
                    raise ValueError("模拟库版本不同，已停止；不会覆盖旧数据")
                actual = {name: connection.scalar(select(func.count()).select_from(table))
                          for name, table in metadata.tables.items() if name in existing}
                if actual != previous["counts"]:
                    raise ValueError("已有模拟数据数量发生变化，已停止；不会自动补写或覆盖")
                return {"created": False, "dataset_label": "SIMULATED", "counts": actual,
                        "end_date": str(previous["end_date"]), **schema_status()}
        for table in metadata.tables.values():
            if table.name in existing and connection.scalar(select(func.count()).select_from(table)):
                raise ValueError(f"业务表 {table.name} 已有数据，已停止；不会覆盖或混入模拟数据")
    records = build_records(end_date, days)
    metadata.create_all(engine)
    manifest.create(engine, checkfirst=True)
    counts = {}
    with engine.begin() as connection:
        # Recheck after DDL; a second seeder must never append duplicates.
        if connection.scalar(select(func.count()).select_from(manifest)):
            raise ValueError("已有其他初始化记录，已停止")
        for name, table in metadata.tables.items():
            if connection.scalar(select(func.count()).select_from(table)):
                raise ValueError(f"业务表 {name} 已有数据，已停止")
            rows = records[name]
            if rows:
                # Explicit NULLs allow rows with different optional fields in one batch.
                prepared = [{col.name: row.get(col.name) for col in table.columns} for row in rows]
                connection.execute(table.insert(), prepared)
            counts[name] = len(rows)
        connection.execute(manifest.insert(), {"id": 1, "seed_version": SEED_VERSION,
            "catalog_hash": catalog_hash(), "dataset_label": "SIMULATED", "end_date": end_date or date.today() - timedelta(days=1),
            "counts": counts, "assumptions": ASSUMPTIONS})
    return {"created": True, "dataset_label": "SIMULATED", "counts": counts, **schema_status()}
