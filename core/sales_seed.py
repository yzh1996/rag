"""只在明确执行时写入销量样例，不覆盖已有数据。"""
from datetime import date, datetime, timedelta, timezone
from random import Random

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.db_models import LiveSales, VideoSales


def sample_sales(end_date: date | None = None):
    """固定随机种子，两个渠道共用一个连续30天的快照。"""
    now = datetime.now(timezone.utc)
    end_date = end_date or (now.astimezone(timezone(timedelta(hours=8))).date() - timedelta(days=1))
    common = {"period_start": end_date - timedelta(days=29), "period_end": end_date, "updated_at": now}
    random = Random(20261016)
    live = [{**common, "room_id": f"R{index + 1:03d}", "anchor_id": f"A{index % 19 + 1:03d}",
        "sales_quantity": random.randint(100, 9000)} for index in range(40)]
    video_anchors = list(range(1, 19)) + [20]
    video = [{**common, "video_id": f"V{index + 1:03d}",
        "anchor_id": f"A{video_anchors[index % len(video_anchors)]:03d}",
        "sales_quantity": random.randint(20, 3000)} for index in range(120)]
    # 零销量和并列销量；A019仅有直播，A020仅有视频。
    for rows in (live, video):
        rows[0]["sales_quantity"] = 0
        rows[1]["sales_quantity"] = 0
        rows[2]["sales_quantity"] = 500
        rows[3]["sales_quantity"] = 500
    return live, video


def seed_sales(engine, end_date: date | None = None):
    """事务内检查并插入样例；任一销量表非空就停止。"""
    live, video = sample_sales(end_date)
    LiveSales.__table__.create(engine, checkfirst=True)
    VideoSales.__table__.create(engine, checkfirst=True)
    with Session(engine) as db, db.begin():
        if db.scalar(select(func.count()).select_from(LiveSales)) or db.scalar(select(func.count()).select_from(VideoSales)):
            raise ValueError("销量表已有数据，已停止；不会覆盖或重复插入。")
        db.execute(LiveSales.__table__.insert(), live)
        db.execute(VideoSales.__table__.insert(), video)
    totals = {}
    for row in live + video:
        totals[row["anchor_id"]] = totals.get(row["anchor_id"], 0) + row["sales_quantity"]
    return {"live_rows": len(live), "video_rows": len(video), "anchors": len(totals),
        "period_start": str(live[0]["period_start"]), "period_end": str(live[0]["period_end"]),
        "live_total": sum(row["sales_quantity"] for row in live),
        "video_total": sum(row["sales_quantity"] for row in video),
        "top5": sorted(totals.items(), key=lambda item: (-item[1], item[0]))[:5]}
