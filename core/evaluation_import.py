"""评测测试集文件导入。"""
import csv
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".json"}


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _list_value(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [item.strip() for item in map(str, value) if item.strip()]

    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
    return [item.strip() for item in text.replace("，", ",").split(",") if item.strip()]


def _rows_from_file(filename: str, content: bytes) -> list[dict[str, Any]]:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"不支持的文件格式，请使用：{supported}")

    if extension == ".csv":
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("CSV文件必须使用UTF-8编码") from exc
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames or "question" not in reader.fieldnames:
            raise ValueError("文件必须包含 question 列")
        return [dict(row) for row in reader]

    if extension == ".json":
        try:
            payload = json.loads(content.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("JSON文件格式不正确") from exc
        rows = payload.get("items") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("JSON文件必须是数组，或包含 items 数组")
        return [row for row in rows if isinstance(row, dict)]

    try:
        dataframe = pd.read_excel(io.BytesIO(content), engine="openpyxl")
    except Exception as exc:
        raise ValueError(f"Excel文件读取失败：{exc}") from exc
    if "question" not in dataframe.columns:
        raise ValueError("文件必须包含 question 列")
    return dataframe.fillna("").to_dict(orient="records")


def parse_eval_cases(filename: str, content: bytes, dataset_name: str) -> list[dict[str, Any]]:
    """读取并校验测试集文件，返回可写入 EvalCase 的数据。"""
    rows = _rows_from_file(filename, content)
    if not rows:
        raise ValueError("文件中没有可导入的测试题")

    cases: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 2):
        question = _text(row.get("question"))
        if not question:
            raise ValueError(f"第 {index} 行的 question 不能为空")
        behavior = _text(row.get("expected_behavior")) or "answer"
        if behavior not in {"answer", "refuse", "clarify"}:
            raise ValueError(f"第 {index} 行的预期行为必须是 answer、refuse 或 clarify")
        history = row.get("history") or []
        access = row.get("access_context") or {}
        if isinstance(history, str):
            history = json.loads(history)
        if isinstance(access, str):
            access = json.loads(access)
        if not isinstance(history, list) or any(not isinstance(item, dict) or item.get("role") not in {"user", "human", "assistant", "ai"} or not isinstance(item.get("content"), str) for item in history):
            raise ValueError(f"第 {index} 行的历史对话格式不正确")
        if not isinstance(access, dict):
            raise ValueError(f"第 {index} 行的测试身份格式不正确")
        if set(access) - {"username", "department"} or any(not isinstance(value, str) for value in access.values()):
            raise ValueError(f"第 {index} 行只能设置本租户普通用户名和部门")
        item = {
                "dataset_name": dataset_name,
                "question": question,
                "expected_answer": _text(row.get("expected_answer")) or None,
                "expected_sources": _list_value(row.get("expected_sources")),
                "history": history,
                "category": _text(row.get("category")) or None,
                "tags": _list_value(row.get("tags")),
            }
        if "expected_behavior" in row:
            item["expected_behavior"] = behavior
        if "require_citations" in row:
            item["require_citations"] = _text(row.get("require_citations")).lower() in {"true", "1", "yes"}
        if "access_context" in row:
            item["access_context"] = access
        cases.append(item)
    return cases
