import json
from io import BytesIO

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.database import Base
from core.evaluation_import import parse_eval_cases
from core.phase1_services import create_eval_cases, list_eval_cases, list_eval_datasets


def test_parse_csv_eval_cases():
    content = (
        "question,expected_answer,expected_sources,tags\n"
        "年假如何计算？,按工龄计算,leave-policy|制度\n"
    ).replace("leave-policy|制度", "leave-policy,制度").encode("utf-8")

    cases = parse_eval_cases("sample.csv", content, "人事制度")

    assert cases == [
        {
            "dataset_name": "人事制度",
            "question": "年假如何计算？",
            "expected_answer": "按工龄计算",
            "expected_sources": ["leave-policy"],
            "history": [],
            "category": None,
            "tags": ["制度"],
        }
    ]


def test_parse_json_eval_cases():
    content = json.dumps(
        {"items": [{"question": "产品如何使用？", "tags": ["产品", "培训"]}]},
        ensure_ascii=False,
    ).encode("utf-8")

    cases = parse_eval_cases("sample.json", content, "产品问答")

    assert cases[0]["dataset_name"] == "产品问答"
    assert cases[0]["tags"] == ["产品", "培训"]


def test_parse_xlsx_eval_cases():
    buffer = BytesIO()
    pd.DataFrame([{"question": "如何申请？", "expected_answer": "提交申请"}]).to_excel(
        buffer, index=False
    )

    cases = parse_eval_cases("sample.xlsx", buffer.getvalue(), "流程集")

    assert cases[0]["question"] == "如何申请？"
    assert cases[0]["expected_answer"] == "提交申请"


def test_imported_cases_are_grouped_by_dataset(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'evaluation-test.db'}")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    db = session_factory()
    try:
        create_eval_cases(
            db,
            [
                {
                    "dataset_name": "制度集",
                    "question": "问题一",
                    "expected_sources": [],
                    "tags": [],
                },
                {
                    "dataset_name": "产品集",
                    "question": "问题二",
                    "expected_sources": [],
                    "tags": [],
                },
            ],
        )

        assert len(list_eval_cases(db, dataset_name="制度集")) == 1
        assert list_eval_datasets(db) == [
            {"name": "产品集", "case_count": 1},
            {"name": "制度集", "case_count": 1},
        ]
    finally:
        db.close()


def test_parse_eval_cases_requires_question():
    with pytest.raises(ValueError, match="question 不能为空"):
        content = json.dumps([{"expected_answer": "answer"}]).encode("utf-8")
        parse_eval_cases("sample.json", content, "测试集")
