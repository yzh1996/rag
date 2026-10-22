"""构建/更新独立的表结构与指标规则索引，不修改业务数据库。"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.sales_schema_retriever import build_sales_schema_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建可信表结构索引；只索引项目维护的结构与规则。")
    parser.parse_args()
    try:
        print(json.dumps(build_sales_schema_index(), ensure_ascii=False, indent=2))
    except Exception as exc:
        parser.exit(1, f"表结构索引构建失败（{type(exc).__name__}），请检查目录和向量模型。\n")
