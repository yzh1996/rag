"""Create the isolated 90-day synthetic commerce database."""
import argparse
from datetime import date
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.commerce_demo_data import seed_demo_commerce_data
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="创建隔离的模拟经营数据库，不改动项目问答数据库")
    parser.add_argument("--end-date", type=date.fromisoformat, help="模拟数据结束日，默认今天")
    args = parser.parse_args()
    print(seed_demo_commerce_data(end_date=args.end_date))
