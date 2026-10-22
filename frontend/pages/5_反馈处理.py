import streamlit as st

from frontend.api_client import API_BASE, client, render_auth

st.set_page_config(page_title="反馈处理", layout="wide")
render_auth()
st.title("反馈处理")


def call(method, path, **kwargs):
    try:
        response = client.request(method, API_BASE + path, timeout=15, **kwargs)
        if not response.ok:
            st.error(response.json().get("detail", "请求失败"))
            return None
        return response.json()
    except client.RequestException:
        st.error("无法连接服务")
        return None


status = st.selectbox("处理状态", ["全部", "open", "pending_handoff", "in_progress", "resolved", "closed"])
data = call("GET", "/feedback", params={} if status == "全部" else {"status": status})
items = (data or {}).get("items", [])
if data is not None and not items:
    st.info("暂无反馈")
if items:
    selected = st.selectbox("反馈记录", range(len(items)), format_func=lambda index: f"{items[index]['question'][:60]} · {items[index]['status']}")
    item = items[selected]
    st.subheader("用户问题")
    st.write(item["question"])
    st.subheader("原始回答")
    st.write(item["answer"])
    st.write("评价：", item["rating"], "分类：", item["category"] or "其他")
    st.write("反馈说明：", item["comment"] or "无")
    st.write("人工处理：", "已申请" if item["handoff_requested"] else "未申请")
    with st.form("review_" + item["id"]):
        options = ["open", "pending_handoff", "in_progress", "resolved", "closed"]
        new_status = st.selectbox("更新状态", options, index=options.index(item["status"]))
        assignee = st.text_input("处理人", value=item["assignee"] or "")
        notes = st.text_area("处理结论", value=item["resolution_notes"] or "")
        correction = st.text_area("确认后的正确答案", value=item["corrected_answer"] or "")
        if st.form_submit_button("保存处理结果"):
            if call("PATCH", "/feedback/" + item["id"], json={"status": new_status, "assignee": assignee or None,
                "resolution_notes": notes or None, "corrected_answer": correction or None}):
                st.rerun()
    if item["promoted_case_id"]:
        st.success("已加入回归测试集：" + item["promoted_case_id"])
    else:
        with st.form("promote_" + item["id"]):
            dataset = st.text_input("回归测试集", value="feedback-regression")
            expected = st.text_area("人工确认的参考答案", value=item["corrected_answer"] or "")
            if st.form_submit_button("加入回归测试集"):
                if call("POST", "/feedback/" + item["id"] + "/promote", json={"dataset_name": dataset, "expected_answer": expected}):
                    st.rerun()
