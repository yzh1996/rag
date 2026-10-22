"""Isolated synthetic commerce dataset used to exercise the assistant workflow."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
import random
import sqlite3

from config.settings import settings

ANCHORS = [
    ("A001", "陈洁KiKi", "美妆个护", 186000),
    ("A002", "美妆阿沐", "美妆个护", 92000),
    ("A003", "日常好物安安", "家居日用", 143000),
    ("A004", "小美吃货", "食品饮料", 215000),
    ("A005", "穿搭小青", "服饰穿搭", 128000),
    ("A006", "生活研究所", "家居日用", 305000),
]
PRODUCTS = [
    ("P001", "温和洁面套装", "美妆个护", "清研", 129.0),
    ("P002", "补水面膜组合", "美妆个护", "清研", 99.0),
    ("P003", "防晒乳双支装", "美妆个护", "日光实验室", 159.0),
    ("P004", "家用收纳套组", "家居日用", "简居", 89.0),
    ("P005", "轻音加湿器", "家居日用", "简居", 229.0),
    ("P006", "厨房清洁组合", "家居日用", "净物", 69.0),
    ("P007", "坚果分享礼盒", "食品饮料", "山野集", 118.0),
    ("P008", "低糖燕麦组合", "食品饮料", "谷物日记", 79.0),
    ("P009", "轻薄防晒外套", "服饰穿搭", "云衣", 189.0),
    ("P010", "通勤基础针织衫", "服饰穿搭", "云衣", 219.0),
    ("P011", "头皮护理套装", "美妆个护", "植萃日记", 179.0),
    ("P012", "桌面氛围灯", "家居日用", "简居", 139.0),
]
ANCHOR_PRODUCTS = {
    "A001": ["P001", "P002", "P003", "P011"],
    "A002": ["P001", "P003", "P011"],
    "A003": ["P004", "P005", "P006", "P012"],
    "A004": ["P007", "P008", "P002"],
    "A005": ["P009", "P010", "P003"],
    "A006": ["P004", "P005", "P012"],
}
ANCHOR_SCALE = {"A001": 1.0, "A002": 0.72, "A003": 0.86, "A004": 1.18, "A005": 0.79, "A006": 1.32}


def _database_path(path=None):
    return Path(path or settings.COMMERCE_DEMO_DATABASE_PATH).resolve()


def _connect(path=None, readonly=False):
    database_path = _database_path(path)
    if readonly:
        if not database_path.is_file():
            raise FileNotFoundError("演示经营数据库尚未创建，请运行 python scripts/seed_commerce_demo.py")
        connection = sqlite3.connect(database_path.as_uri() + "?mode=ro", uri=True, timeout=2)
    else:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(database_path, timeout=2)
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def _readonly_connection(path=None):
    connection = _connect(path, readonly=True)
    try:
        yield connection
    finally:
        connection.close()


def seed_demo_commerce_data(path=None, *, end_date: date | None = None) -> dict:
    """Create a separate 90-day demo database once; never overwrite existing rows."""
    end_date = end_date or date.today()
    start_date = end_date - timedelta(days=89)
    connection = _connect(path)
    try:
        connection.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS demo_anchors (
                anchor_id TEXT PRIMARY KEY, anchor_name TEXT NOT NULL, main_category TEXT NOT NULL,
                follower_count INTEGER NOT NULL CHECK (follower_count >= 0), dataset_label TEXT NOT NULL DEFAULT 'SIMULATED'
            );
            CREATE TABLE IF NOT EXISTS demo_products (
                product_id TEXT PRIMARY KEY, product_name TEXT NOT NULL, category TEXT NOT NULL,
                brand TEXT NOT NULL, price REAL NOT NULL CHECK (price >= 0), dataset_label TEXT NOT NULL DEFAULT 'SIMULATED'
            );
            CREATE TABLE IF NOT EXISTS demo_live_daily (
                id INTEGER PRIMARY KEY, anchor_id TEXT NOT NULL REFERENCES demo_anchors(anchor_id),
                product_id TEXT NOT NULL REFERENCES demo_products(product_id), room_id TEXT NOT NULL,
                live_date TEXT NOT NULL, sales_quantity INTEGER NOT NULL CHECK (sales_quantity >= 0),
                gmv REAL NOT NULL CHECK (gmv >= 0), viewers INTEGER NOT NULL CHECK (viewers >= 0),
                UNIQUE (anchor_id, product_id, live_date, room_id)
            );
            CREATE TABLE IF NOT EXISTS demo_video_daily (
                id INTEGER PRIMARY KEY, anchor_id TEXT NOT NULL REFERENCES demo_anchors(anchor_id),
                product_id TEXT NOT NULL REFERENCES demo_products(product_id), video_id TEXT NOT NULL,
                publish_date TEXT NOT NULL, sales_quantity INTEGER NOT NULL CHECK (sales_quantity >= 0),
                gmv REAL NOT NULL CHECK (gmv >= 0), plays INTEGER NOT NULL CHECK (plays >= 0),
                UNIQUE (anchor_id, product_id, publish_date, video_id)
            );
            CREATE INDEX IF NOT EXISTS ix_demo_live_anchor_date ON demo_live_daily(anchor_id, live_date);
            CREATE INDEX IF NOT EXISTS ix_demo_video_anchor_date ON demo_video_daily(anchor_id, publish_date);
            CREATE INDEX IF NOT EXISTS ix_demo_live_product ON demo_live_daily(product_id);
            CREATE INDEX IF NOT EXISTS ix_demo_video_product ON demo_video_daily(product_id);
        """)
        existing = connection.execute("SELECT COUNT(*) FROM demo_anchors").fetchone()[0]
        if existing:
            if existing != len(ANCHORS):
                raise ValueError("演示数据库数据不完整；为避免覆盖已有数据，请先检查数据库文件")
            return demo_data_summary(connection)
        connection.execute("BEGIN")
        connection.executemany("INSERT INTO demo_anchors(anchor_id, anchor_name, main_category, follower_count) VALUES (?, ?, ?, ?)", ANCHORS)
        connection.executemany("INSERT INTO demo_products(product_id, product_name, category, brand, price) VALUES (?, ?, ?, ?, ?)", PRODUCTS)
        product_info = {item[0]: item for item in PRODUCTS}
        live_rows, video_rows = [], []
        rng = random.Random(20261017)
        for anchor_index, (anchor_id, _name, _category, _followers) in enumerate(ANCHORS):
            scale = ANCHOR_SCALE[anchor_id]
            for day_index in range(90):
                current = start_date + timedelta(days=day_index)
                if (day_index + anchor_index) % 4 in {0, 2}:
                    room_id = f"R{anchor_id[1:]}-{current.strftime('%Y%m%d')}"
                    for product_index, product_id in enumerate(ANCHOR_PRODUCTS[anchor_id]):
                        if (day_index + product_index) % 3 == 0:
                            continue
                        quantity = max(0, int((14 + (day_index * 7 + product_index * 11 + anchor_index * 5) % 52) * scale + rng.randint(-4, 8)))
                        price = product_info[product_id][4]
                        viewers = int((900 + (day_index * 101 + product_index * 47) % 5400) * scale)
                        live_rows.append((anchor_id, product_id, room_id, current.isoformat(), quantity, round(quantity * price, 2), viewers))
                if (day_index + 2 * anchor_index) % 3 != 1:
                    for product_index, product_id in enumerate(ANCHOR_PRODUCTS[anchor_id]):
                        if (day_index + product_index + anchor_index) % 4 != 0:
                            continue
                        quantity = max(0, int((4 + (day_index * 5 + product_index * 13 + anchor_index * 3) % 24) * scale + rng.randint(-2, 4)))
                        price = product_info[product_id][4]
                        plays = int((500 + (day_index * 89 + product_index * 29) % 8800) * scale)
                        video_id = f"V{anchor_id[1:]}-{day_index:02d}-{product_index:02d}"
                        video_rows.append((anchor_id, product_id, video_id, current.isoformat(), quantity, round(quantity * price, 2), plays))
        connection.executemany("INSERT INTO demo_live_daily(anchor_id, product_id, room_id, live_date, sales_quantity, gmv, viewers) VALUES (?, ?, ?, ?, ?, ?, ?)", live_rows)
        connection.executemany("INSERT INTO demo_video_daily(anchor_id, product_id, video_id, publish_date, sales_quantity, gmv, plays) VALUES (?, ?, ?, ?, ?, ?, ?)", video_rows)
        connection.commit()
        return {**demo_data_summary(connection), "created": True}
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def demo_data_summary(connection=None, path=None) -> dict:
    should_close = connection is None
    connection = connection or _connect(path, readonly=True)
    try:
        anchors = connection.execute("SELECT COUNT(*) FROM demo_anchors").fetchone()[0]
        products = connection.execute("SELECT COUNT(*) FROM demo_products").fetchone()[0]
        live = connection.execute("SELECT COUNT(*), MIN(live_date), MAX(live_date) FROM demo_live_daily").fetchone()
        video = connection.execute("SELECT COUNT(*), MIN(publish_date), MAX(publish_date) FROM demo_video_daily").fetchone()
        starts = [value for value in (live[1], video[1]) if value]
        ends = [value for value in (live[2], video[2]) if value]
        return {"database": str(_database_path(path)), "dataset_label": "SIMULATED", "anchors": anchors,
            "products": products, "live_rows": live[0], "video_rows": video[0],
            "period_start": min(starts) if starts else None, "period_end": max(ends) if ends else None}
    finally:
        if should_close:
            connection.close()


def find_anchors(name: str, path=None) -> list[dict]:
    normalized = (name or "").strip().casefold()
    if not normalized:
        return []
    with _readonly_connection(path) as connection:
        rows = connection.execute("SELECT anchor_id, anchor_name, main_category, follower_count FROM demo_anchors ORDER BY anchor_id").fetchall()
    exact = [row for row in rows if row["anchor_name"].casefold() == normalized or row["anchor_id"].casefold() == normalized]
    matches = exact or [row for row in rows if normalized in row["anchor_name"].casefold() or row["anchor_name"].casefold() in normalized]
    return [dict(row) for row in matches[:10]]



def find_anchor_mentions(text: str, path=None) -> list[dict]:
    """Find exact synthetic account names mentioned in a current question or recent history."""
    with _readonly_connection(path) as connection:
        rows = connection.execute("SELECT anchor_id, anchor_name, main_category, follower_count FROM demo_anchors ORDER BY anchor_id").fetchall()
    value = (text or "").casefold()
    return [dict(row) for row in rows if row["anchor_name"].casefold() in value or row["anchor_id"].casefold() in value]


def _validate_query_window(start_date: str, end_date: str) -> None:
    start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    if start > end:
        raise ValueError("开始日期不能晚于结束日期")
    if (end - start).days + 1 > max(1, settings.COMMERCE_DEMO_MAX_PERIOD_DAYS):
        raise ValueError("查询时间范围超过演示数据工具允许的最大天数")



def _validate_data_coverage(connection, table: str, date_column: str, start_date: str, end_date: str) -> None:
    row = connection.execute(f"SELECT MIN({date_column}) AS first_date, MAX({date_column}) AS last_date FROM {table}").fetchone()
    if not row or not row["first_date"] or start_date < row["first_date"] or end_date > row["last_date"]:
        first = row["first_date"] if row and row["first_date"] else "未知"
        last = row["last_date"] if row and row["last_date"] else "未知"
        raise ValueError(f"演示数据当前仅覆盖{first}至{last}，无法完整回答指定时间范围")


def analyze_anchor_live(anchor_id: str, start_date: str, end_date: str, path=None) -> dict:
    _validate_query_window(start_date, end_date)
    with _readonly_connection(path) as connection:
        _validate_data_coverage(connection, "demo_live_daily", "live_date", start_date, end_date)
        anchor = connection.execute("SELECT anchor_name, main_category FROM demo_anchors WHERE anchor_id = ?", (anchor_id,)).fetchone()
        if not anchor:
            raise ValueError("没有找到这个主播，请核对主播名称或ID")
        summary = connection.execute("""SELECT COUNT(DISTINCT room_id) AS live_count,
            COUNT(DISTINCT live_date) AS active_days, COALESCE(SUM(sales_quantity), 0) AS sales_quantity,
            COALESCE(SUM(gmv), 0) AS gmv, COALESCE(SUM(viewers), 0) AS viewers
            FROM demo_live_daily WHERE anchor_id = ? AND live_date BETWEEN ? AND ?""", (anchor_id, start_date, end_date)).fetchone()
        categories = connection.execute("""SELECT p.category, SUM(l.sales_quantity) AS sales_quantity,
            SUM(l.gmv) AS gmv FROM demo_live_daily l JOIN demo_products p USING(product_id)
            WHERE l.anchor_id = ? AND l.live_date BETWEEN ? AND ?
            GROUP BY p.category ORDER BY sales_quantity DESC, p.category""", (anchor_id, start_date, end_date)).fetchall()
        top_products = connection.execute("""SELECT p.product_id, p.product_name, p.category,
            SUM(l.sales_quantity) AS sales_quantity, ROUND(SUM(l.gmv), 2) AS gmv
            FROM demo_live_daily l JOIN demo_products p USING(product_id)
            WHERE l.anchor_id = ? AND l.live_date BETWEEN ? AND ?
            GROUP BY p.product_id ORDER BY sales_quantity DESC, p.product_id LIMIT 5""", (anchor_id, start_date, end_date)).fetchall()
    return {"anchor_id": anchor_id, "anchor_name": anchor["anchor_name"], "main_category": anchor["main_category"],
        "period_start": start_date, "period_end": end_date, "live_count": summary["live_count"],
        "active_days": summary["active_days"], "sales_quantity": summary["sales_quantity"],
        "gmv": round(summary["gmv"], 2), "viewers": summary["viewers"],
        "categories": [dict(row) for row in categories], "top_products": [dict(row) for row in top_products], "dataset_label": "SIMULATED"}


def analyze_anchor_video(anchor_id: str, start_date: str, end_date: str, path=None) -> dict:
    _validate_query_window(start_date, end_date)
    with _readonly_connection(path) as connection:
        _validate_data_coverage(connection, "demo_video_daily", "publish_date", start_date, end_date)
        anchor = connection.execute("SELECT anchor_name, main_category FROM demo_anchors WHERE anchor_id = ?", (anchor_id,)).fetchone()
        if not anchor:
            raise ValueError("没有找到这个主播，请核对主播名称或ID")
        summary = connection.execute("""SELECT COUNT(DISTINCT video_id) AS video_count,
            COALESCE(SUM(sales_quantity), 0) AS sales_quantity, COALESCE(SUM(gmv), 0) AS gmv,
            COALESCE(SUM(plays), 0) AS plays FROM demo_video_daily
            WHERE anchor_id = ? AND publish_date BETWEEN ? AND ?""", (anchor_id, start_date, end_date)).fetchone()
        categories = connection.execute("""SELECT p.category, SUM(v.sales_quantity) AS sales_quantity,
            SUM(v.gmv) AS gmv FROM demo_video_daily v JOIN demo_products p USING(product_id)
            WHERE v.anchor_id = ? AND v.publish_date BETWEEN ? AND ?
            GROUP BY p.category ORDER BY sales_quantity DESC, p.category""", (anchor_id, start_date, end_date)).fetchall()
        top_products = connection.execute("""SELECT p.product_id, p.product_name, p.category,
            SUM(v.sales_quantity) AS sales_quantity, ROUND(SUM(v.gmv), 2) AS gmv
            FROM demo_video_daily v JOIN demo_products p USING(product_id)
            WHERE v.anchor_id = ? AND v.publish_date BETWEEN ? AND ?
            GROUP BY p.product_id ORDER BY sales_quantity DESC, p.product_id LIMIT 5""", (anchor_id, start_date, end_date)).fetchall()
    return {"anchor_id": anchor_id, "anchor_name": anchor["anchor_name"], "main_category": anchor["main_category"],
        "period_start": start_date, "period_end": end_date, "video_count": summary["video_count"],
        "sales_quantity": summary["sales_quantity"], "gmv": round(summary["gmv"], 2), "plays": summary["plays"],
        "categories": [dict(row) for row in categories], "top_products": [dict(row) for row in top_products], "dataset_label": "SIMULATED"}


def analyze_anchor_products(anchor_id: str, start_date: str, end_date: str, path=None) -> dict:
    _validate_query_window(start_date, end_date)
    with _readonly_connection(path) as connection:
        live_period = connection.execute("SELECT MIN(live_date) AS first_date, MAX(live_date) AS last_date FROM demo_live_daily").fetchone()
        video_period = connection.execute("SELECT MIN(publish_date) AS first_date, MAX(publish_date) AS last_date FROM demo_video_daily").fetchone()
        first_date = max(live_period["first_date"], video_period["first_date"])
        last_date = min(live_period["last_date"], video_period["last_date"])
        if start_date < first_date or end_date > last_date:
            raise ValueError(f"演示数据当前仅覆盖{first_date}至{last_date}，无法完整回答指定时间范围")
        anchor = connection.execute("SELECT anchor_name FROM demo_anchors WHERE anchor_id = ?", (anchor_id,)).fetchone()
        if not anchor:
            raise ValueError("没有找到这个主播，请核对主播名称或ID")
        rows = connection.execute("""WITH channel_sales AS (
            SELECT product_id, sales_quantity, gmv FROM demo_live_daily
            WHERE anchor_id = ? AND live_date BETWEEN ? AND ?
            UNION ALL
            SELECT product_id, sales_quantity, gmv FROM demo_video_daily
            WHERE anchor_id = ? AND publish_date BETWEEN ? AND ?
        ) SELECT p.product_id, p.product_name, p.category, p.brand, p.price,
            COALESCE(SUM(c.sales_quantity), 0) AS sales_quantity, ROUND(COALESCE(SUM(c.gmv), 0), 2) AS gmv
            FROM channel_sales c JOIN demo_products p USING(product_id)
            GROUP BY p.product_id ORDER BY sales_quantity DESC, p.product_id LIMIT 12""",
            (anchor_id, start_date, end_date, anchor_id, start_date, end_date)).fetchall()
    products = [dict(row) for row in rows]
    total = sum(item["sales_quantity"] for item in products)
    for item in products:
        item["sales_share"] = round(item["sales_quantity"] / total, 4) if total else 0.0
    categories = {}
    for item in products:
        row = categories.setdefault(item["category"], {"category": item["category"], "sales_quantity": 0, "gmv": 0.0})
        row["sales_quantity"] += item["sales_quantity"]
        row["gmv"] += item["gmv"]
    category_rows = sorted(categories.values(), key=lambda item: (-item["sales_quantity"], item["category"]))
    for row in category_rows:
        row["gmv"] = round(row["gmv"], 2)
        row["sales_share"] = round(row["sales_quantity"] / total, 4) if total else 0.0
    top3_share = sum(item["sales_quantity"] for item in products[:3]) / total if total else 0.0
    return {"anchor_id": anchor_id, "anchor_name": anchor["anchor_name"], "period_start": start_date,
        "period_end": end_date, "total_sales_quantity": total, "top3_product_share": round(top3_share, 4),
        "categories": category_rows, "products": products, "dataset_label": "SIMULATED"}


def calculate_channel_metrics(live: dict | None, video: dict | None, products: dict | None = None) -> dict:
    comparable = live is not None and video is not None
    live_sales = int(live["sales_quantity"]) if live else None
    video_sales = int(video["sales_quantity"]) if video else None
    total = live_sales + video_sales if comparable else None
    return {"total_sales_quantity": total,
        "live_sales_share": round(live_sales / total, 4) if comparable and total else None,
        "video_sales_share": round(video_sales / total, 4) if comparable and total else None,
        "live_video_gap": live_sales - video_sales if comparable else None,
        "top3_product_share": (products or {}).get("top3_product_share"),
        "comparison_status": "complete" if comparable else "incomplete_channel_data",
        "dataset_label": "SIMULATED"}


def compare_anchor_category(anchor_id: str, start_date: str, end_date: str, path=None) -> dict:
    """Compare one demo creator with other synthetic accounts in the same main category."""
    _validate_query_window(start_date, end_date)
    with _readonly_connection(path) as connection:
        live_period = connection.execute("SELECT MIN(live_date) AS first_date, MAX(live_date) AS last_date FROM demo_live_daily").fetchone()
        video_period = connection.execute("SELECT MIN(publish_date) AS first_date, MAX(publish_date) AS last_date FROM demo_video_daily").fetchone()
        first_date = max(live_period["first_date"], video_period["first_date"])
        last_date = min(live_period["last_date"], video_period["last_date"])
        if start_date < first_date or end_date > last_date:
            raise ValueError(f"演示数据当前仅覆盖{first_date}至{last_date}，无法完整回答指定时间范围")
        anchor = connection.execute("SELECT anchor_name, main_category FROM demo_anchors WHERE anchor_id = ?", (anchor_id,)).fetchone()
        if not anchor:
            raise ValueError("没有找到这个主播，请核对主播名称或ID")
        peers = connection.execute("""WITH channel_rows AS (
            SELECT anchor_id, product_id, sales_quantity FROM demo_live_daily WHERE live_date BETWEEN ? AND ?
            UNION ALL
            SELECT anchor_id, product_id, sales_quantity FROM demo_video_daily WHERE publish_date BETWEEN ? AND ?
        ) SELECT a.anchor_id, a.anchor_name, COALESCE(SUM(c.sales_quantity), 0) AS total_sales_quantity,
            COUNT(DISTINCT c.product_id) AS product_count
            FROM demo_anchors a LEFT JOIN channel_rows c ON c.anchor_id = a.anchor_id
            WHERE a.main_category = ? AND a.anchor_id != ?
            GROUP BY a.anchor_id, a.anchor_name ORDER BY total_sales_quantity DESC, a.anchor_id""",
            (start_date, end_date, start_date, end_date, anchor["main_category"], anchor_id)).fetchall()
        own = connection.execute("""WITH channel_rows AS (
            SELECT sales_quantity FROM demo_live_daily WHERE anchor_id = ? AND live_date BETWEEN ? AND ?
            UNION ALL
            SELECT sales_quantity FROM demo_video_daily WHERE anchor_id = ? AND publish_date BETWEEN ? AND ?
        ) SELECT COALESCE(SUM(sales_quantity), 0) FROM channel_rows""",
            (anchor_id, start_date, end_date, anchor_id, start_date, end_date)).fetchone()[0]
    peer_rows = [dict(row) for row in peers]
    peer_values = sorted(item["total_sales_quantity"] for item in peer_rows)
    midpoint = len(peer_values) // 2
    peer_median = (peer_values[midpoint] if len(peer_values) % 2 else
        (peer_values[midpoint - 1] + peer_values[midpoint]) / 2) if peer_values else None
    return {"anchor_id": anchor_id, "anchor_name": anchor["anchor_name"], "category": anchor["main_category"],
        "period_start": start_date, "period_end": end_date, "own_total_sales_quantity": own,
        "peer_count": len(peer_rows), "peer_median_sales_quantity": peer_median,
        "peers": peer_rows, "sample_size_warning": "模拟同类账号数量较少，不能代表行业整体" if len(peer_rows) < 5 else None,
        "dataset_label": "SIMULATED"}
