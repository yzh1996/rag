"""对冻结测试题执行评测，不写入用户聊天历史。"""
from datetime import datetime, timezone

from ragas import EvaluationDataset, evaluate
from ragas.metrics import answer_correctness, answer_relevancy, context_precision, context_recall, faithfulness
from sqlalchemy import delete

from config.settings import settings
from core.auth import AuthUser
from core.database import init_db, session_scope
from core.db_models import EvalResult, EvalRun, IndexVersion
from core.evaluation_contract import contract_checks, fingerprint, runtime_config
from core.rag_chain import get_rag_chain
from core.release_service import artifact_hash, release_store


def run_evaluation(run_id, limit=0, user=None):
    from core.evaluation_service import _case_passed, _mark_failed, _safe_metrics, _update_progress
    init_db()
    try:
        with session_scope() as db:
            run = db.get(EvalRun, run_id)
            if not run:
                raise ValueError("评测记录不存在")
            if run.status == "completed":
                return
            config = dict(run.config or {})
            cases = config.get("case_snapshot")
            tenant_id = run.tenant_id
            if not cases or fingerprint(cases) != config.get("dataset_revision"):
                raise ValueError("评测缺少有效的冻结测试集，请新建评测")
            if config.get("runtime") != runtime_config():
                raise ValueError("评测创建后配置已变化，请新建评测")
            if limit > 0:
                cases = cases[:limit]
            run.status = "running"
            run.total_cases = len(cases)
            run.passed_cases = 0
            run.metrics = {}
            db.execute(delete(EvalResult).where(EvalResult.run_id == run_id))
        chain = get_rag_chain()
        index_id = config.get("index_id")
        if index_id:
            with session_scope() as db:
                item = db.get(IndexVersion, index_id)
                if not item or item.tenant_id != tenant_id:
                    raise ValueError("评测索引版本不存在")
                if config.get("artifact_hash") != item.config.get("artifact_hash") or artifact_hash(release_store(item)) != config.get("artifact_hash"):
                    raise ValueError("候选索引与创建评测时不一致")
        outputs, groups = [], {True: [], False: []}
        for index, case in enumerate(cases):
            _update_progress(run_id, phase="answering", completed_cases=index, total_cases=len(cases), current_case=index + 1)
            actor = user or AuthUser(username="evaluation", tenant_id=tenant_id)
            access = case.get("access_context") or {}
            if access:
                # 测试身份只能模拟本租户普通用户，不能从测试文件提升为管理员。
                actor = AuthUser(username=access.get("username", actor.username), department=access.get("department"),
                    tenant_id=tenant_id, role="user", authenticated=True)
            if actor.tenant_id != tenant_id:
                raise ValueError("评测身份与测试集租户不一致")
            options = {"history": case.get("history") or [], "persist": False}
            if index_id:
                options["index_id"] = index_id
            result = chain.ask(case["question"], session_id=f"eval:{run_id}:{case['id']}", user=actor, **options)
            contexts = result.get("retrieved_contexts", [])
            checks = contract_checks(case, result)
            outputs.append({"case": case, "result": result, "contexts": contexts, "checks": checks, "scores": {}})
            if case.get("expected_behavior", "answer") == "answer" and all(checks.values()):
                sample = {"user_input": case["question"], "response": result["answer"], "retrieved_contexts": contexts}
                has_reference = bool(case.get("expected_answer"))
                if has_reference:
                    sample["reference"] = case["expected_answer"]
                groups[has_reference].append((index, sample))
        _update_progress(run_id, phase="metrics", completed_cases=len(cases), total_cases=len(cases), current_case=None)
        for has_reference, entries in groups.items():
            if not entries:
                continue
            metrics = [faithfulness, answer_relevancy]
            if has_reference:
                metrics += [context_precision, context_recall, answer_correctness]
            evaluation = evaluate(EvaluationDataset.from_list([sample for _, sample in entries]), metrics=metrics,
                llm=chain.llm, embeddings=chain.retriever.vector_store.embedding_model, raise_exceptions=False)
            if len(evaluation.scores) != len(entries):
                raise ValueError("评测评分数量不完整")
            for (index, _), scores in zip(entries, evaluation.scores):
                outputs[index]["scores"] = _safe_metrics(scores)
        passed_count = 0
        collected = {}
        with session_scope() as db:
            run = db.get(EvalRun, run_id)
            for output in outputs:
                case, scores, checks = output["case"], output["scores"], output["checks"]
                passed = all(checks.values())
                if case.get("expected_behavior", "answer") == "answer":
                    passed = passed and _case_passed(scores, bool(case.get("expected_answer")))
                    if case.get("expected_answer"):
                        passed = passed and (scores.get("answer_correctness") or 0) >= settings.EVAL_MIN_ANSWER_CORRECTNESS
                passed_count += int(passed)
                for key, value in scores.items():
                    if value is not None:
                        collected.setdefault(key, []).append(value)
                db.add(EvalResult(run_id=run_id, case_id=case["id"], answer=output["result"].get("answer"),
                    retrieved_sources=output["contexts"], metrics={**scores, "checks": checks,
                        "sources": output["result"].get("sources", []), "case_snapshot": case}, passed=passed,
                    error=output["result"].get("error")))
            run.status = "completed"
            run.passed_cases = passed_count
            run.completed_at = datetime.now(timezone.utc)
            complete_suite = len(cases) == len(config["case_snapshot"])
            run.metrics = {key: sum(values) / len(values) for key, values in collected.items()}
            run.metrics = {**run.metrics, "quality_gate_passed": passed_count == len(cases) and complete_suite,
                "complete_suite": complete_suite, "dataset_revision": config["dataset_revision"],
                "runtime_fingerprint": fingerprint(config["runtime"])}
        _update_progress(run_id, phase="completed", completed_cases=len(cases), total_cases=len(cases), current_case=None)
    except Exception as exc:
        _mark_failed(run_id, str(exc))
