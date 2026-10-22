from frontend.api_client import client as requests, API_BASE, render_auth
import json
import streamlit as st



st.set_page_config(page_title="RAG评测中心", page_icon="📊", layout="wide")
render_auth()
st.title("📊 RAG评测中心")
st.caption("管理测试集、运行离线评测并查看逐题结果")


def api_request(method: str, path: str, **kwargs):
    try:
        response = requests.request(method, f"{API_BASE}{path}", timeout=kwargs.pop("timeout", 15), **kwargs)
        if response.status_code >= 400:
            st.error(f"接口请求失败 ({response.status_code}): {response.text}")
            return None
        return response.json()
    except requests.RequestException as exc:
        st.error(f"无法连接到后端服务: {exc}")
        return None


def load_cases(active_only: bool = False):
    data = api_request("GET", "/evaluation/cases", params={"active_only": str(active_only).lower()})
    return data.get("items", []) if data else []


def load_datasets():
    data = api_request("GET", "/evaluation/datasets")
    return data.get("items", []) if data else []


def load_runs():
    data = api_request("GET", "/evaluation/runs", params={"limit": 50})
    return data.get("items", []) if data else []


PHASE_LABELS = {
    "answering": "正在生成答案",
    "metrics": "正在计算 Ragas 指标",
    "completed": "已完成",
    "failed": "执行失败",
}


def render_run_progress(run):
    progress = run.get("progress") or {}
    completed = int(progress.get("completed_cases") or 0)
    total = int(progress.get("total_cases") or run.get("total_cases") or 0)
    phase = PHASE_LABELS.get(progress.get("phase"), "等待执行")
    current = progress.get("current_case")
    if total:
        st.progress(min(completed / total, 1.0))
        current_text = f"，当前第 {current} 题" if current else ""
        st.caption(f"{phase}：已完成 {completed} / {total} 题{current_text}")
    else:
        st.caption(phase)


@st.fragment(run_every="2s")
def render_run_history():
    if st.button("刷新运行状态", key="refresh_runs"):
        st.rerun()
    runs = load_runs()
    if not runs:
        st.info("暂无评测运行记录。")
    for run in runs:
        status = run.get("status", "unknown")
        status_label = {"pending": "等待执行", "running": "评测中", "completed": "已完成", "failed": "失败"}.get(status, status)
        st.write(
            f"`{run.get('id')}` · **{run.get('dataset_name')}** · {status_label} · "
            f"题数 {run.get('total_cases', 0)} · 通过 {run.get('passed_cases', 0)}"
        )
        if status in {"pending", "running"}:
            render_run_progress(run)


@st.fragment(run_every="2s")
def render_run_detail(selected_run):
    detail = api_request("GET", f"/evaluation/runs/{selected_run}")
    if not detail:
        return

    status = detail.get("status", "unknown")
    status_label = {"pending": "等待执行", "running": "评测中", "completed": "已完成", "failed": "失败"}.get(status, status)
    st.write(f"状态：**{status_label}**")
    if status in {"pending", "running"}:
        render_run_progress(detail)
        st.info("评测正在后台执行，页面会自动刷新进度。")
    if status == "failed":
        st.error((detail.get("config") or {}).get("error", "评测执行失败"))

    if st.button("刷新当前结果", key="refresh_result_detail"):
        st.rerun()

    metrics = detail.get("metrics") or {}
    if metrics:
        numeric = {name: value for name, value in metrics.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
        metric_columns = st.columns(min(4, len(numeric))) if numeric else []
        for column, (name, value) in zip(metric_columns, numeric.items()):
            display_value = f"{float(value):.4f}"
            column.metric(name, display_value)
        st.write("发布门禁：", "通过" if metrics.get("quality_gate_passed") else "未通过")
        st.caption("测试集版本：" + str(metrics.get("dataset_revision", "历史记录未固定")))

    st.divider()
    st.subheader("逐题结果")
    results = detail.get("results") or []
    if not results:
        st.info("当前运行还没有逐题结果。")
    for index, result in enumerate(results, 1):
        result_status = "通过" if result.get("passed") else "未完成或失败"
        with st.expander(f"第 {index} 题 · {result_status} · {result.get('question') or result.get('case_id')}"):
            st.write(f"**答案：** {result.get('answer') or '无'}")
            st.write(f"**召回内容：** {len(result.get('retrieved_sources') or [])} 条")
            st.json(result.get("metrics") or {})
            for source_index, source in enumerate(result.get("retrieved_sources") or [], 1):
                st.text(f"来源片段 {source_index}: {source[:500]}")
            if result.get("error"):
                st.error(result["error"])


tab_cases, tab_run, tab_results = st.tabs(["测试集管理", "运行评测", "评测结果"])

with tab_cases:
    st.subheader("导入测试集")
    import_dataset_name = st.text_input("测试集名称", value="default", key="import_dataset_name")
    import_file = st.file_uploader(
        "上传测试集文件（支持 CSV、XLSX、JSON）",
        type=["csv", "xlsx", "json"],
        key="eval_dataset_file",
    )
    if st.button("上传并导入测试集", type="primary", key="import_eval_dataset"):
        if not import_dataset_name.strip():
            st.warning("测试集名称不能为空")
        elif not import_file:
            st.warning("请先选择测试集文件")
        else:
            data = api_request(
                "POST",
                "/evaluation/cases/import",
                files={"file": (import_file.name, import_file.getvalue())},
                data={"dataset_name": import_dataset_name.strip()},
                timeout=60,
            )
            if data:
                st.success(
                    f"测试集“{data['dataset_name']}”导入成功，共 {data['imported_count']} 道题"
                )
                st.rerun()

    st.divider()
    st.subheader("新增测试题")
    with st.form("create_eval_case"):
        dataset_name = st.text_input("测试集名称", value="default")
        question = st.text_area("问题", placeholder="例如：员工入职满一年有几天年假？")
        expected_answer = st.text_area("参考答案", placeholder="填写人工确认的标准答案")
        expected_sources = st.text_input("期望来源", placeholder="多个来源用逗号分隔")
        category = st.text_input("分类", placeholder="例如：制度、产品、流程")
        tags = st.text_input("标签", placeholder="多个标签用逗号分隔")
        behavior = st.selectbox("期望行为", ["answer", "refuse", "clarify"], format_func=lambda v: {"answer": "回答问题", "refuse": "拒绝无依据或越权请求", "clarify": "请用户补充问题"}[v])
        citations = st.checkbox("要求答案引用来源")
        previous_question = st.text_input("上一轮问题（多轮测试可填）")
        previous_answer = st.text_area("上一轮回答")
        case_user = st.text_input("模拟提问用户名（权限测试可填）")
        case_department = st.text_input("模拟提问部门")
        submitted = st.form_submit_button("保存测试题", type="primary")

    if submitted:
        if not question.strip():
            st.warning("问题不能为空")
        else:
            payload = {
                "dataset_name": dataset_name.strip() or "default",
                "question": question.strip(),
                "expected_answer": expected_answer.strip() or None,
                "expected_sources": [item.strip() for item in expected_sources.split(",") if item.strip()],
                "category": category.strip() or None,
                "tags": [item.strip() for item in tags.split(",") if item.strip()],
                "expected_behavior": behavior, "require_citations": citations,
                "history": [{"role": "user", "content": previous_question}, {"role": "assistant", "content": previous_answer}] if previous_question else [],
                "access_context": {key: value for key, value in (("username", case_user), ("department", case_department)) if value},
            }
            if api_request("POST", "/evaluation/cases", json=payload):
                st.success("测试题已保存")
                st.rerun()

    st.divider()
    show_all = st.checkbox("显示已禁用测试题", value=False)
    cases = load_cases(active_only=not show_all)
    st.metric("当前测试题数量", len(cases))

    if not cases:
        st.info("暂无测试题，请先添加问题和参考答案。")
    for case in cases:
        status = "启用" if case.get("active") else "已禁用"
        with st.expander(
            f"{status} · {case.get('dataset_name', 'default')} · {case.get('question', '')}"
        ):
            st.write(f"**参考答案：** {case.get('expected_answer') or '未填写'}")
            st.write(f"**期望来源：** {', '.join(case.get('expected_sources') or []) or '未填写'}")
            st.write(f"**分类：** {case.get('category') or '未分类'}")
            st.write(f"**标签：** {', '.join(case.get('tags') or []) or '无'}")
            if case.get("active"):
                if st.button("禁用此题", key=f"disable_case_{case['id']}"):
                    if api_request("PATCH", f"/evaluation/cases/{case['id']}", json={"active": False}):
                        st.success("测试题已禁用")
                        st.rerun()
            else:
                if st.button("重新启用", key=f"enable_case_{case['id']}"):
                    if api_request("PATCH", f"/evaluation/cases/{case['id']}", json={"active": True}):
                        st.success("测试题已启用")
                        st.rerun()

with tab_run:
    st.subheader("启动离线评测")
    datasets = load_datasets()
    dataset_options = [item["name"] for item in datasets]
    if dataset_options:
        dataset_name = st.selectbox("选择测试集", dataset_options, key="run_dataset_name")
        active_cases = [
            case for case in load_cases(active_only=True)
            if case.get("dataset_name", "default") == dataset_name
        ]
    else:
        dataset_name = "default"
        active_cases = []
        st.info("暂无可用测试集，请先导入或新增测试题。")
    st.write(f"当前测试集启用测试题：{len(active_cases)} 条")
    versions = (api_request("GET", "/index/versions") or {}).get("items", [])
    options = [None] + [item["id"] for item in versions if item["status"] in {"candidate", "canary", "published", "archived"}]
    index_id = st.selectbox("评测索引", options, format_func=lambda value: "当前线上索引（仅作诊断）" if value is None else next(v["version"] + " · " + v["status"] for v in versions if v["id"] == value))
    limit = st.number_input(
        "最多评测题数（0表示全部）",
        min_value=0,
        max_value=1000,
        value=0,
        step=1,
    )
    if st.button("开始评测", type="primary", disabled=not active_cases):
        data = api_request(
            "POST",
            "/evaluation/runs",
            json={"dataset_name": dataset_name.strip() or "default", "limit": int(limit), "config": {"index_id": index_id}},
            timeout=30,
        )
        if data:
            st.session_state.selected_eval_run = data["id"]
            st.success(f"评测已提交，运行编号：{data['id']}")

    st.divider()
    st.subheader("评测运行历史")
    render_run_history()

with tab_results:
    st.subheader("评测结果")
    runs = load_runs()
    if not runs:
        st.info("暂无评测运行记录。")
    else:
        run_options = [run["id"] for run in runs]
        default_run = st.session_state.get("selected_eval_run", run_options[0])
        default_index = run_options.index(default_run) if default_run in run_options else 0
        selected_run = st.selectbox("选择评测运行", run_options, index=default_index)
        if st.button("查看结果", type="primary"):
            st.session_state.selected_eval_run = selected_run

        render_run_detail(selected_run)
        comparison = st.selectbox("对比另一轮评测", [None] + [value for value in run_options if value != selected_run])
        if comparison and st.button("比较两轮结果"):
            left = api_request("GET", "/evaluation/runs/" + selected_run)
            right = api_request("GET", "/evaluation/runs/" + comparison)
            if left and right:
                if left.get("config", {}).get("dataset_revision") != right.get("config", {}).get("dataset_revision"):
                    st.warning("两轮测试题或参考答案不同，不能直接比较通过率")
                else:
                    st.dataframe([{"运行": value["id"], "索引": value.get("config", {}).get("index_id") or "线上诊断",
                        "通过": value["passed_cases"], "总题数": value["total_cases"], "门禁通过": value.get("metrics", {}).get("quality_gate_passed", False)} for value in (left, right)], hide_index=True)
                    before = {result["case_id"]: result for result in left.get("results", [])}
                    changed = [{"问题": result.get("question"), "第一轮通过": before[result["case_id"]]["passed"],
                        "第二轮通过": result["passed"]} for result in right.get("results", []) if result["case_id"] in before
                        and result["passed"] != before[result["case_id"]]["passed"]]
                    if changed:
                        st.dataframe(changed, hide_index=True)
                    else:
                        st.info("两轮逐题通过情况没有变化")
