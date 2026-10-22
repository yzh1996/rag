"""知识版本、后台任务和候选索引的操作入口。"""
import streamlit as st

from frontend.api_client import API_BASE, client


STATUS_LABELS = {"preview": "等待分块确认", "pending": "等待入库", "processing": "正在入库",
    "candidate": "待发布", "published": "已发布", "archived": "历史版本", "deprecated": "已下线",
    "failed": "处理失败", "building": "正在构建", "canary": "灰度发布中", "rolled_back": "已回退"}
STRATEGY_LABELS = {"recursive": "递归字符切分", "structure": "结构感知切分", "parent_child": "父子分块",
    "semantic": "语义切分", "fixed": "固定长度切分"}


def _document_label(document):
    version = f"当前生效第{document['current_version']}版" if document["current_version"] else "尚未正式发布"
    return f"{document['original_filename']} · {STATUS_LABELS.get(document['status'], document['status'])} · {version}"


def call(method, path, **kwargs):
    try:
        response = client.request(method, API_BASE + path, timeout=30, **kwargs)
        if not response.ok:
            st.error(response.json().get("detail", "请求失败"))
            return None
        return response.json()
    except (client.RequestException, ValueError):
        st.error("无法连接服务或服务响应异常")
        return None


def render_documents():
    data = call("GET", "/documents")
    documents = (data or {}).get("items", [])
    if not documents:
        st.info("暂无可访问文档")
        return
    st.caption("本页用于查看文档版本、更新文件和管理权限。当前文件已确认的分块方式无需重复选择；审核与发布请进入“知识库审核与发布”。")
    selected = st.selectbox("文档", documents, format_func=_document_label)
    detail = call("GET", "/documents/" + selected["id"])
    if detail:
        st.dataframe([{"文档版本": v.get("version"), "状态": STATUS_LABELS.get(v.get("status"), v.get("status")),
            "分块方式": STRATEGY_LABELS.get(v.get("chunking_strategy"), v.get("chunking_strategy")),
            "生效时间": v.get("effective_at"), "失效时间": v.get("expired_at")} for v in detail["versions"]], hide_index=True)
    with st.expander("更新已有文件（可选）", expanded=False):
        st.caption("仅在文件内容有更新时使用。下方分块方式只用于本次新上传的更新文件，不会修改当前版本已确认的分块结果；提交后仍需审核、评测和发布。")
        with st.form("update_document"):
            upload = st.file_uploader("上传更新后的文件", type=["pdf", "docx", "doc", "txt", "md", "xlsx", "xls"])
            strategy = st.selectbox("更新文件的分块方式", list(STRATEGY_LABELS), format_func=STRATEGY_LABELS.get)
            if st.form_submit_button("提交更新文件") and upload:
                if call("POST", f"/documents/{selected['id']}/versions", files={"file": (upload.name, upload.getvalue(), upload.type)},
                    data={"category": selected["category"], "chunking_strategy": strategy}):
                    st.success("更新文件已提交入库任务，完成后请到“知识库审核与发布”处理")
    left, right = st.columns(2)
    if left.button("下线所选文档", disabled=selected["status"] == "deprecated"):
        if call("POST", f"/documents/{selected['id']}/deprecate"):
            st.rerun()
    if right.button("恢复所选文档", disabled=selected["status"] != "deprecated"):
        if call("POST", f"/documents/{selected['id']}/restore"):
            st.rerun()
    with st.expander("文档读取权限（管理员）"):
        scope = selected.get("permission_scope") or {}
        st.caption("留空表示该项不限；多项限制需要同时满足。扩大原知识分组权限需要重新上传、评测和发布。")
        with st.form("permission_scope"):
            users = st.text_input("允许的用户名（逗号分隔）", value=",".join(scope.get("users", [])))
            roles = st.text_input("允许的角色（逗号分隔）", value=",".join(scope.get("roles", [])))
            departments = st.text_input("允许的部门（逗号分隔）", value=",".join(scope.get("departments", [])))
            if st.form_submit_button("保存读取权限"):
                updated = {"tenant_ids": scope.get("tenant_ids") or [selected["tenant_id"]]}
                for key, text in (("users", users), ("roles", roles), ("departments", departments)):
                    if text.strip():
                        updated[key] = [part.strip() for part in text.split(",") if part.strip()]
                if call("PATCH", f"/documents/{selected['id']}/permissions", json={"permission_scope": updated}):
                    st.success("权限已保存")


def render_tasks():
    if st.button("刷新任务状态"):
        st.rerun()
    for item in (call("GET", "/jobs") or {}).get("items", []):
        with st.expander(f"文档入库 · {item['id']} · {item['status']}"):
            st.write("重试次数：", item["retry_count"])
            if item.get("error_message"):
                st.error(item["error_message"])
    tasks = (call("GET", "/jobs/queue/list") or {}).get("items", [])
    for item in tasks:
        with st.expander(f"后台任务 · {item['kind']} · {item['status']} · {item['id']}"):
            st.write("已尝试次数：", item["attempts"])
            if item.get("error_message"):
                st.error(item["error_message"])
            if item["status"] == "failed" and st.button("重试任务", key="retry_" + item["id"]):
                if call("POST", "/jobs/queue/" + item["id"] + "/retry"):
                    st.rerun()


def render_indexes():
    st.caption("这里审核和发布供问答使用的一整套知识库版本，可包含多份已入库文档，无需重新选择文档的分块方式。")
    st.caption("顺序：构建待发布版本 → 人工审核 → 在评测中心选择该版本并运行完整测试集 → 灰度或正式发布。首次发布需正式发布。")
    if st.button("构建待发布知识库版本", type="primary"):
        if call("POST", "/index/versions/snapshot", json={}):
            st.success("构建任务已提交，可在后台任务页查看进度")
    if st.button("刷新知识库版本"):
        st.rerun()
    data = call("GET", "/index/versions")
    versions = (data or {}).get("items", [])
    if not versions:
        return
    item = st.selectbox("知识库版本", versions, format_func=lambda v: f"{v['version']} · {STATUS_LABELS.get(v['status'], v['status'])}")
    st.write("包含文档数：", len(item.get("config", {}).get("manifest", {})))
    if item.get("config", {}).get("error"):
        st.error(item["config"]["error"])
    if item["status"] == "candidate":
        with st.form("review_index"):
            notes = st.text_area("人工审核结论", placeholder="确认候选知识内容、权限与来源")
            if st.form_submit_button("确认审核") and call("POST", f"/index/versions/{item['id']}/review", json={"notes": notes}):
                st.success("审核已保存")
    if item["status"] in {"candidate", "canary"}:
        runs = (call("GET", "/evaluation/runs") or {}).get("items", [])
        passed = []
        for run in runs:
            if run.get("metrics", {}).get("quality_gate_passed"):
                detail = call("GET", "/evaluation/runs/" + run["id"])
                if detail and detail.get("config", {}).get("index_id") == item["id"]:
                    passed.append(run["id"])
        if passed:
            with st.form("publish_index"):
                run = st.selectbox("此版本的合格评测", passed)
                percent = st.number_input("灰度百分比（0表示正式发布）", 0, 50, 0)
                if st.form_submit_button("发布所选版本") and call("POST", f"/index/versions/{item['id']}/publish", json={"eval_run_id": run, "canary_percent": percent}):
                    st.success("发布已完成")
        else:
            st.info("此版本暂无合格的完整评测，请到评测中心运行")
    if item.get("published_at") and item["status"] != "published":
        if st.button("回退到所选正式版本") and call("POST", f"/index/versions/{item['id']}/rollback"):
            st.success("回退已完成")


def render_management():
    section = st.radio("管理事项", ["文档版本与权限", "后台任务", "知识库审核与发布"], horizontal=True)
    {"文档版本与权限": render_documents, "后台任务": render_tasks, "知识库审核与发布": render_indexes}[section]()
