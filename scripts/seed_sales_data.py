"""生成近30天销量样例：python scripts/seed_sales_data.py [--end-date YYYY-MM-DD]。"""
import argparse
from datetime import date
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.database import engine
from core.sales_seed import seed_sales


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="写入两张销量表的模拟数据；已有数据时停止。")
    parser.add_argument("--end-date", type=date.fromisoformat, help="统计截止日期，默认北京时间昨天")
    args = parser.parse_args()
    try:
        print(json.dumps(seed_sales(engine, args.end_date), ensure_ascii=False, indent=2))
    except ValueError as exc:
        parser.exit(1, str(exc) + "\n")
