"""运行第一阶段RAG离线评测。

用法：
    python scripts/run_ragas_eval.py

评测在离线执行，不会影响在线问答请求。需要数据库中至少有一条测试集数据，
并且当前LLM和Embedding配置可用。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings
from core.database import init_db, session_scope
from core.db_models import EvalRun
from core.evaluation_service import execute_eval_run
from core.phase1_services import create_eval_run, list_eval_cases
from core.auth import AuthUser, actor_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行RAG离线评测")
    parser.add_argument("--dataset", default="default", help="评测集名称")
    parser.add_argument("--limit", type=int, default=0, help="最多评测多少条，0表示全部")
    parser.add_argument("--tenant", default="default", help="测试集租户")
    parser.add_argument("--username", default="evaluation", help="本地离线评测身份")
    parser.add_argument("--index-id", default=None, help="指定候选索引ID，不填仅诊断当前线上版本")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    actor = AuthUser(username=args.username, tenant_id=args.tenant)
    init_db()
    with session_scope() as db:
        cases = list_eval_cases(db, dataset_name=args.dataset, tenant_id=args.tenant)
        if args.limit > 0:
            cases = cases[: args.limit]
        if not cases:
            print("没有可用测试集，请先通过页面或 POST /api/v1/evaluation/cases 添加测试问题。")
            return 1
        run = create_eval_run(
            db,
            args.dataset,
            {
                "llm_provider": settings.LLM_PROVIDER,
                "llm_model": (
                    settings.OLLAMA_MODEL_NAME
                    if settings.LLM_PROVIDER == "ollama"
                    else settings.SILICONFLOW_MODEL_NAME
                    if settings.LLM_PROVIDER == "siliconflow"
                    else settings.OPENAI_MODEL_NAME
                ),
                "embedding_model": settings.EMBEDDING_MODEL_NAME,
                "index_id": args.index_id, "actor": actor_payload(actor), "limit": args.limit,
            },
            tenant_id=args.tenant,
        )
        run_id = run.id

    execute_eval_run(run_id, args.limit, actor)
    with session_scope() as db:
        run = db.get(EvalRun, run_id)
        payload = {
            "run_id": run_id,
            "status": run.status if run else "unknown",
            "metrics": run.metrics if run else {},
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
