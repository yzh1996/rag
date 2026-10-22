"""Create dictionary-backed synthetic data in the explicitly configured local MySQL."""
from pathlib import Path
import argparse
from datetime import date
import json
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from config.settings import settings
from core.business_seed import seed_business_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create-database", action="store_true", help="仅在目标本地数据库不存在时创建")
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()
    url = make_url(settings.DATABASE_URL)
    if url.get_backend_name() != "mysql":
        raise ValueError("请在 .env 配置 MYSQL_USER、MYSQL_PASSWORD、MYSQL_DATABASE 后再运行；当前不是MySQL")
    if url.host not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("本脚本仅为本机MySQL造数，不允许连接远程地址")
    if not url.database or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", url.database):
        raise ValueError("目标数据库名称无效")
    options = {"connect_args": {"connect_timeout": 5, "read_timeout": 30, "write_timeout": 30}, "pool_pre_ping": True}
    if args.create_database:
        admin = create_engine(url.set(database=""), **options)
        try:
            with admin.connect() as connection:
                connection.exec_driver_sql(f"CREATE DATABASE IF NOT EXISTS `{url.database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_bin")
        finally:
            admin.dispose()
    engine = create_engine(url, **options)
    try:
        result = seed_business_data(engine, args.end_date, args.days)
        print(json.dumps({"database": url.database, **result}, ensure_ascii=False, indent=2))
    finally:
        engine.dispose()


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        # Never print a database URL, password, or the DBAPI connection arguments.
        print(f"MySQL初始化失败（{type(exc).__name__}），请检查本机连接、权限和空库状态。", file=sys.stderr)
        sys.exit(1)
