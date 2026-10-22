"""资料上传、阶段结果、内容确认和分块预览。"""
import json

import streamlit as st

from frontend.api_client import API_BASE, client
from frontend.content_compare import PREVIEW_LIMIT, render_content_compare, render_chunk_compare

STATUS_LABELS = {"checking": "正在检查", "queued": "等待内容处理", "processing": "正在处理内容",
    "ready": "等待内容确认", "chunking": "正在生成分块", "preview": "等待分块确认",
    "submitted": "已提交入库", "candidate": "入库完成，待审核", "published": "文档版本已发布",
    "failed": "处理未通过", "ingestion_failed": "向量化入库失败"}
STAGE_LABELS = {"pending": "等待", "running": "处理中", "passed": "通过", "skipped": "无需执行", "failed": "未通过"}
STRATEGIES = {"递归字符切分": "recursive", "固定长度切分": "fixed", "结构感知切分": "structure",
    "语义切分": "semantic", "父子分块": "parent_child"}


def _request(method, path, **kwargs):
    if path.startswith("/upload/") and "upload_tenant_id" in st.session_state:
        kwargs["params"] = {**kwargs.get("params", {}), "tenant_id": st.session_state["upload_tenant_id"]}
    try:
        response = client.request(method, API_BASE + path, **kwargs)
        if response.status_code not in {200, 202}:
            try:
                body = response.json()
                if path == "/upload/tasks" and body.get("status") == "failed" and body.get("record_saved") is False:
                    return body
                detail = body.get("detail")
            except (ValueError, AttributeError):
                detail = None
            if response.status_code == 401:
                st.warning("请先使用有效身份登录，系统才能确定所属公司、部门和上传权限。")
            else:
                st.error(detail if isinstance(detail, str) else "操作未完成，请检查填写的信息或联系管理员。")
            return None
        return response.json()
    except client.RequestException:
        st.error("暂时无法连接服务。已创建的处理记录会保留，可以稍后刷新查看。")
        return None


def _clear_company_state():
    for key in list(st.session_state):
        if key.startswith("upload_") and key != "upload_tenant_id":
            st.session_state.pop(key, None)


def _upload_form(options):
    bases = options.get("knowledge_bases", [])
    departments = options.get("departments", [])
    if not bases or not departments:
        st.warning("当前账号没有可用的部门上传权限，请联系管理员配置财务部或客服部授权。")
        return
    st.subheader("1. 资料上传与归属")
    st.caption("归属部门负责维护资料，可见范围控制谁能查看和用于问答。")
    if options.get("tenant_editable"):
        # 公司共享库允许独立选择资料归属部门与可见范围。
        bases = sorted(bases, key=lambda item: item.get("owner_department") is not None)
        st.caption("当前为免登录管理员模式。公司编号留空使用默认公司；不同公司请填写不同编号。")
    left, right = st.columns(2)
    with left:
        if options.get("tenant_editable"):
            st.text_input("公司 / 租户 ID（可选）", key="upload_tenant_id",
                max_chars=128, on_change=_clear_company_state,
                help="留空归入默认公司。切换公司后，文件选择和处理记录会同步切换。")
        else:
            st.text_input("所属公司（租户）", value=options["tenant_id"], disabled=True)
        selected_id = st.selectbox("目标知识库", [item["id"] for item in bases],
            format_func=lambda value: next(item["name"] for item in bases if item["id"] == value),
            key="upload_target")
    selected = next(item for item in bases if item["id"] == selected_id)
    with right:
        possible = [selected["owner_department"]] if selected.get("owner_department") in departments else departments
        department = st.selectbox("归属部门", possible, key="upload_department")
        visibility = st.selectbox("可见范围", ["继承知识库权限", "仅本人", "指定部门", "指定人员"], key="upload_visibility")
    allowed_departments, allowed_users = [], []
    if visibility == "指定部门":
        choices = [name for name in selected.get("read_departments", departments) if name in departments]
        allowed_departments = st.multiselect("允许查看的部门", choices, key="upload_read_departments")
    elif visibility == "指定人员":
        allowed_users = st.multiselect("允许查看的人员", options.get("users", []), key="upload_read_users")
    st.caption("文档权限只能在目标知识库的允许范围内进一步缩小；不会因选择部门而获得额外权限。")
    with st.form("upload_documents", clear_on_submit=False):
        files = st.file_uploader("选择资料（支持多文件）",
            type=["pdf", "docx", "doc", "txt", "md", "xlsx", "xls"], accept_multiple_files=True,
            key="upload_files_" + options["tenant_id"])
        st.caption(f"单文件上限 {options.get('max_upload_mb', 30)}MB。每个文件独立检查，一个失败不影响其他文件。")
        left, right = st.columns(2)
        with left:
            category = st.selectbox("文档分类", ["general", "管理制度", "技术文档", "产品手册", "培训材料", "assistant_tools", "其他"],
                key="upload_category")
        with right:
            owners = options.get("business_owners", [options["username"]])
            owner = st.selectbox("业务负责人", owners, index=owners.index(options["username"]) if options["username"] in owners else 0,
                key="upload_owner", format_func=lambda name: "本地管理员" if options.get("tenant_editable") and name == options["username"] else name)
        source = st.text_input("资料来源说明（可选）", key="upload_source",
            help="可填写原始系统、网址或业务编号；采集方式自动记录为手工上传。")
        title = st.text_input("文档标题（可选）", key="upload_title",
            help="留空使用文件名；多文件上传时分别使用各自文件名。")
        submitted = st.form_submit_button("上传并开始检查", type="primary", use_container_width=True)
    if submitted:
        if not files:
            st.warning("请先选择需要上传的文件。")
            return
        if visibility == "指定部门" and not allowed_departments or visibility == "指定人员" and not allowed_users:
            st.warning("请先选择允许查看的部门或人员。")
            return
        if len(files) > 1 and title.strip():
            st.warning("多文件上传请将统一标题留空，系统会分别使用文件名，避免资料混淆。")
            return
        metadata = {"tenant_id": options["tenant_id"], "owner_department": department, "business_owner": owner, "category": category,
            "source_reference": source, "title": title,
            "visibility": {"继承知识库权限": "inherit", "仅本人": "self", "指定部门": "departments", "指定人员": "users"}[visibility],
            "allowed_departments": allowed_departments, "allowed_users": allowed_users}
        accepted, failed = 0, 0
        with st.spinner("正在上传并检查权限、文件和重复提交…"):
            for file in files:
                result = _request("POST", "/upload/tasks",
                    files={"file": (file.name, file.getvalue())},
                    data={"knowledge_base_id": selected_id, "metadata": json.dumps(metadata, ensure_ascii=False)},
                    timeout=120)
                if result and result.get("status") != "failed":
                    accepted += 1
                else:
                    failed += 1
                    if result:
                        error = next(item for item in result["stages"] if item["status"] == "failed")
                        st.error(f"{file.name}：{error['label']}未通过。{error['detail']}")
                        st.info(error["next_action"])
        st.info(f"本次已受理 {accepted} 个文件，未通过或未提交 {failed} 个。详细结果在下面的处理记录中查看。")


def _chunk_parameters(task_id):
    label = st.selectbox("分块方式", list(STRATEGIES), key="upload_strategy_" + task_id)
    strategy, params = STRATEGIES[label], {}
    with st.expander("高级分块参数"):
        if strategy in {"recursive", "fixed", "structure"}:
            params["chunk_size"] = st.number_input("分块大小", 100, 5000, 800 if strategy == "structure" else 500,
                50, key="upload_size_" + task_id + strategy)
            params["chunk_overlap"] = st.number_input("重叠大小", 0, 1000, 50, 10, key="upload_overlap_" + task_id + strategy)
            if strategy == "structure":
                params["preserve_headers"] = st.checkbox("保留标题", value=True, key="upload_headers_" + task_id)
        elif strategy == "semantic":
            st.caption("语义分块需要调用已配置的模型服务，处理时间和费用取决于资料内容。")
            params["breakpoint_threshold"] = st.slider("语义阈值", 50, 99, 75, key="upload_threshold_" + task_id)
            params["min_chunk_size"] = st.number_input("最小片段大小", 50, 2000, 200, 50, key="upload_min_" + task_id)
            params["max_chunk_size"] = st.number_input("最大片段大小", 100, 5000, 800, 50, key="upload_max_" + task_id)
        else:
            params["parent_size"] = st.number_input("父块大小", 300, 10000, 1000, 100, key="upload_parent_" + task_id)
            params["child_size"] = st.number_input("子块大小", 100, 5000, 300, 50, key="upload_child_" + task_id)
            params["child_overlap"] = st.number_input("子块重叠", 0, 1000, 50, 10, key="upload_child_overlap_" + task_id)
    return strategy, params


def _content_review(task):
    task_id = task["id"]
    st.subheader("3. 内容处理与确认")
    render_content_compare(task_id, task.get("before_content", ""), task.get("after_content", ""))
    st.caption("同步滚动按两侧阅读进度联动；需要单独调整时，关闭上方的“同步滚动”。")
    left, right = st.columns(2)
    for column, field, label in ((left, "before_content", "清洗前（含按需OCR结果）"),
        (right, "after_content", "清洗后")):
        text = task.get(field, "")
        with column:
            if len(text) > PREVIEW_LIMIT:
                st.caption("页面展示前60000个字符，请下载完整文本检查。")
            st.download_button("下载" + label + "完整文本", text.encode("utf-8"),
                file_name=task["filename"] + ("_before.txt" if field == "before_content" else "_after.txt"),
                mime="text/plain", key="upload_download_" + field + task_id)
    chunk_failed = any(item["key"] == "chunks" and item["status"] == "failed" for item in task["stages"])
    if task["status"] in {"ready", "preview"} or task["status"] == "failed" and chunk_failed:
        checked = st.checkbox("我已检查处理后的文字，可以进入分块", key="upload_content_checked_" + task_id)
        strategy, params = _chunk_parameters(task_id)
        if st.button("确认内容并生成分块", disabled=not checked, key="upload_generate_" + task_id, type="primary"):
            if _request("POST", f"/upload/tasks/{task_id}/chunks", json={"strategy": strategy, "params": params}):
                st.session_state.pop("upload_chunks_checked_" + task_id, None)
                st.rerun()


def _remember_selected_task():
    st.session_state["upload_current_task"] = st.session_state["upload_selected_task"]


@st.fragment(run_every="3s")
def _processing_records():
    response = _request("GET", "/upload/tasks", timeout=15)
    if response is None:
        return
    records = response.get("items", [])
    st.subheader("2. 文件处理记录")
    st.caption("处理状态自动刷新。离开页面或服务重启后，可以重新打开查看。")
    if "upload_record_notice" in st.session_state:
        st.success(st.session_state.pop("upload_record_notice"))
    if not records:
        st.session_state.pop("upload_current_task", None)
        st.session_state.pop("upload_selected_task", None)
        st.info("上传文件后，这里会分别显示每一步的结果。")
        return
    st.dataframe([{"文件": item["filename"], "状态": STATUS_LABELS.get(item["status"], item["status"]),
        "归属部门": item.get("metadata", {}).get("owner_department", ""), "上传时间": item["created_at"]}
        for item in records], hide_index=True, use_container_width=True)
    selector, removal = st.columns([4, 1], vertical_alignment="bottom")
    task_ids = [item["id"] for item in records]
    current = st.session_state.get("upload_current_task", st.session_state.get("upload_selected_task"))
    with selector:
        selected = st.selectbox("查看文件处理详情", task_ids,
            format_func=lambda value: next(item["filename"] + " · " + STATUS_LABELS.get(item["status"], item["status"]) +
                " · " + item["created_at"] for item in records if item["id"] == value),
            index=task_ids.index(current) if current in task_ids else 0, key="upload_selected_task",
            on_change=_remember_selected_task)
    # 状态刷新会改变下拉框身份，单独保存任务编号以保持当前文件。
    st.session_state["upload_current_task"] = selected
    task = _request("GET", "/upload/tasks/" + selected, timeout=15)
    if task is None:
        return
    with removal:
        if st.button("× 删除记录", key="upload_delete_" + selected, disabled=not task.get("can_delete", False),
            help="删除当前选中文件的处理记录，关联资料保留。正在处理或等待处理时不可删除。"):
            result = _request("DELETE", "/upload/tasks/" + selected)
            if result:
                for key in list(st.session_state):
                    if key.startswith("upload_") and selected in key:
                        st.session_state.pop(key, None)
                st.session_state.pop("upload_selected_task", None)
                st.session_state.pop("upload_current_task", None)
                st.session_state["upload_record_notice"] = result["message"]
                st.rerun()
    blocked = False
    rows = []
    for stage in task["stages"]:
        state = "前序未通过，未执行" if blocked and stage["status"] == "pending" else STAGE_LABELS.get(stage["status"], stage["status"])
        rows.append({"环节": stage["label"], "状态": state, "说明": stage.get("detail", ""),
            "耗时（秒）": round(stage["duration_ms"] / 1000, 2) if stage.get("duration_ms") is not None else None})
        blocked = blocked or stage["status"] == "failed"
    st.dataframe(rows, hide_index=True, use_container_width=True)
    for stage in task["stages"]:
        if stage["status"] == "failed":
            st.error(f"{task['filename']}：{stage['label']}未通过。{stage.get('detail', '')}")
            st.info(stage.get("next_action") or "请联系管理员处理。")
    with st.expander("文档归属与处理结果"):
        meta = task.get("metadata", {})
        st.write({"所属公司": task["tenant_id"], "归属部门": meta.get("owner_department"),
            "文档标题": meta.get("title"), "业务负责人": meta.get("business_owner"),
            "上传人": meta.get("uploaded_by"), "资料来源": meta.get("source_reference") or "手工上传"})
        for stage in task["stages"]:
            stats = stage.get("stats", {})
            if stage["key"] == "clean" and stats:
                st.caption(f"清洗前 {stats['before_chars']} 字符；清洗后 {stats['after_chars']} 字符；移除 {stats['removed_empty_units']} 个空内容单元。")
    if task.get("can_retry") and st.button("从失败阶段重试", key="upload_retry_" + selected):
        if _request("POST", f"/upload/tasks/{selected}/retry"):
            st.rerun()
    if task.get("can_retry_ingestion") and st.button("重试向量化入库", key="upload_ingest_retry_" + selected):
        if _request("POST", f"/upload/tasks/{selected}/retry-ingestion"):
            st.rerun()
    if task["status"] == "ready":
        st.info("内容处理已完成。请先检查文字，再确认进入分块。")
    if task.get("after_content") and task.get("can_read_content"):
        _content_review(task)
    elif task.get("metadata") and not task.get("can_read_content"):
        st.info("当前阅读权限不允许查看处理后的正文。如需检查内容，请联系管理员。")
    chunks = task.get("chunks", [])
    if chunks and task["status"] == "preview":
        st.subheader("4. 分块预览与确认入库")
        st.caption(f"共 {len(chunks)} 个{'子块' if task.get('strategy') == 'parent_child' else '片段'}。"
            "左侧为本次分块使用的清洗后全文，右侧为完整分块列表；父子分块按父块分组显示。确认后使用本次预览结果入库。")
        render_chunk_compare(selected, task.get("after_content", ""), chunks, task.get("strategy"))
        checked = st.checkbox("我已检查分块结果，确认向量化入库", key="upload_chunks_checked_" + selected)
        if st.button("确认入库", disabled=not checked, type="primary", key="upload_confirm_" + selected):
            if _request("POST", f"/upload/tasks/{selected}/confirm"):
                st.session_state["upload_record_notice"] = f"已确认“{task['filename']}”的分块，已提交后台向量化入库，请查看当前文件的处理进度。"
                st.rerun()
    if task["status"] in {"submitted", "candidate", "published"}:
        st.info("已进入向量化和版本管理流程。审核、评测和发布请进入“版本与发布 → 知识库审核与发布”；上传完成不代表已经可用于问答。")


def render_upload_flow():
    st.caption("资料上传 → 权限 / 文件安全 / 去重 → 解析 / 按需OCR / 清洗 → 内容确认 → 分块预览")
    options = _request("GET", "/upload/options", timeout=15)
    if not options or not options.get("tenant_id"):
        return
    _upload_form(options)
    _processing_records()
