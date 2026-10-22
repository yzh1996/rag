import streamlit as st
from frontend.api_client import client as requests, API_BASE, render_auth
from frontend.qa_stream import read_answer_events
from frontend.sales_query import render_sales_query
from frontend.commerce_report import render_commerce_analysis
from frontend.agent_report import render_agent_report
import json
import uuid
import streamlit.components.v1 as components

# API基础地址


def submit_feedback(interaction_id: str, rating: str, **details) -> bool:
    """提交当前问答的简单反馈。"""
    try:
        response = requests.post(
            f"{API_BASE}/qa/feedback",
            json={"interaction_id": interaction_id, "rating": rating, **details},
            timeout=5,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False

st.set_page_config(page_title="智能问答", page_icon="💬")
render_auth()

st.title("💬 智能问答")
st.caption("支持通用问答、内部知识检索和达人经营分析，支持多轮对话")
with st.expander("销量查询示例"):
    st.write("直播销量最高的五个房间是哪些？")
    st.write("主播 A001 的直播和视频销量分别是多少？")
    st.write("近30天直播加视频销量最高的五个主播是谁？")
    st.write("分析陈洁KiKi近90天直播和视频表现，并给出选品方向。")
    st.caption("新经营分析目前连接本地模拟数据，报告会明确标注模拟数据。")

with st.expander("我的反馈处理结果"):
    if st.button("查询处理进度"):
        try:
            response = requests.get(API_BASE + "/feedback/mine", timeout=10)
            if response.ok:
                items = response.json().get("items", [])
                if not items:
                    st.info("暂无反馈记录")
                for item in items:
                    st.write(item["question"], "·", item["status"])
                    st.write("处理结论：", item.get("resolution_notes") or "待处理")
                    if item.get("corrected_answer") and item["status"] in {"resolved", "closed"}:
                        st.write("确认后的答案：", item["corrected_answer"])
            else:
                st.error("反馈结果查询失败，请检查登录状态")
        except requests.RequestException:
            st.error("无法连接到后端服务")

# 保存会话到 LocalStorage
def save_sessions():
    """保存会话列表和当前会话到浏览器 LocalStorage"""
    if "sessions" in st.session_state and "current_session" in st.session_state:
        components.html(
            f"""
            <script>
                localStorage.setItem('rag_qa_sessions', JSON.stringify({json.dumps(st.session_state.sessions)}));
                localStorage.setItem('rag_qa_current_session', '{st.session_state.current_session}');
            </script>
            """,
            height=0
        )

# 从 URL 参数加载会话数据
def init_sessions_from_url():
    """从 URL 参数初始化会话"""
    url_params = st.query_params
    if 'sessions' in url_params and 'current_session' in url_params:
        try:
            sessions_str = url_params['sessions']
            current_session = url_params['current_session']
            st.session_state.sessions = json.loads(sessions_str)
            st.session_state.current_session = current_session
            # 清除 URL 参数
            st.query_params.clear()
            return True
        except:
            pass
    return False

# 初始化会话状态
if not init_sessions_from_url():
    if "sessions" not in st.session_state:
        saved = []
        try:
            response = requests.get(API_BASE + "/qa/sessions", timeout=5)
            if response.ok:
                saved = response.json().get("items", [])
        except requests.RequestException:
            pass
        if saved:
            st.session_state.sessions = {row["session_id"]: row["name"] for row in saved}
            st.session_state.current_session = saved[0]["session_id"]
        else:
            st.session_state.current_session = str(uuid.uuid4())
            st.session_state.sessions = {st.session_state.current_session: "新会话"}
        save_sessions()
    if "current_session" not in st.session_state:
        st.session_state.current_session = list(st.session_state.sessions.keys())[0]
if "messages" not in st.session_state:
    st.session_state.messages = []
    try:
        response = requests.get(API_BASE + "/qa/history", params={"session_id": st.session_state.current_session}, timeout=5)
        if response.ok:
            st.session_state.messages = response.json().get("history", [])
    except requests.RequestException:
        pass

# 侧边栏会话管理
with st.sidebar:
    st.subheader("📋 会话管理")
    
    # 显示会话列表
    selected_session_id = st.selectbox(
        "选择会话",
        options=list(st.session_state.sessions),
        format_func=lambda value: st.session_state.sessions[value],
        index=list(st.session_state.sessions).index(st.session_state.current_session)
    )
    
    # 切换会话
    if selected_session_id != st.session_state.current_session:
        st.session_state.current_session = selected_session_id
        save_sessions()
        st.session_state.messages = []
        try:
            response = requests.get(API_BASE + "/qa/history", params={"session_id": selected_session_id}, timeout=5)
            if response.ok:
                st.session_state.messages = response.json().get("history", [])
        except requests.RequestException:
            pass
        st.rerun()
    
    # 创建新会话
    if st.button("➕ 创建新会话", use_container_width=True):
        new_session_id = str(uuid.uuid4())
        new_session_name = f"会话 {len(st.session_state.sessions) + 1}"
        st.session_state.sessions[new_session_id] = new_session_name
        st.session_state.current_session = new_session_id
        st.session_state.messages = []
        save_sessions()
        st.rerun()
    
    # 重命名会话
    rename_session = st.text_input("重命名会话", value=st.session_state.sessions[st.session_state.current_session])
    if rename_session and rename_session != st.session_state.sessions[st.session_state.current_session]:
        st.session_state.sessions[st.session_state.current_session] = rename_session
        save_sessions()
        st.rerun()
    
    st.divider()
    
    # 清除当前会话历史
    if st.button("🗑️ 清除对话历史", use_container_width=True):
        st.session_state.messages = []
        try:
            response = requests.post(f"{API_BASE}/qa/clear_memory?session_id={st.session_state.current_session}", timeout=5)
            if response.status_code == 200:
                st.success("对话历史已清除")
        except:
            st.warning("清除记忆失败，请检查API服务")
        st.rerun()
    
    # 显示当前会话信息
    st.caption(f"当前会话: {st.session_state.sessions[st.session_state.current_session]}")
    st.caption("提示：保留最近对话和已确认的任务信息；过期数据会重新查询")

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if (msg.get("query_details") or {}).get("analysis_type") == "agent":
            render_agent_report(msg.get("query_details"))
        elif (msg.get("query_details") or {}).get("analysis_type") in {"creator_commerce", "shop_live"}:
            render_commerce_analysis(msg.get("query_details"))
        else:
            render_sales_query(msg.get("query_details"))
        if msg.get("first_token_ms") is not None:
            st.caption(f"首字耗时：{msg['first_token_ms'] / 1000:.2f} 秒（后台接收至首段正文）")
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 查看引用来源"):
                for src in msg["sources"]:
                    st.caption(f"**{src['source']}** (页码:{src.get('page','-')})")
                    st.text(src["content"][:200] + "..." if len(src.get("content", "")) > 200 else src.get("content", ""))
        if msg.get("role") == "assistant" and msg.get("interaction_id"):
            with st.expander("反馈与人工处理"):
                with st.form(f"feedback_details_{msg['interaction_id']}"):
                    rating = st.radio("评价", ["有帮助", "没帮助"], horizontal=True)
                    category = st.selectbox("问题类型", ["其他", "检索不准", "答案错误", "引用问题", "权限问题", "响应缓慢"])
                    resolved = st.radio("问题是否解决", ["未解决", "已解决", "暂不确定"], horizontal=True)
                    comment = st.text_area("反馈说明")
                    correction = st.text_area("建议的正确答案")
                    handoff = st.checkbox("申请人工处理")
                    if st.form_submit_button("提交反馈"):
                        if submit_feedback(msg["interaction_id"], "positive" if rating == "有帮助" else "negative",
                            category=category, comment=comment, corrected_answer=correction or None,
                            is_resolved=True if resolved == "已解决" else False if resolved == "未解决" else None,
                            handoff_requested=handoff):
                            msg["feedback"] = "submitted"
                            st.success("反馈已保存，人工处理申请会进入待处理列表" if handoff else "反馈已保存")
                        else:
                            st.error("反馈提交失败，请检查身份或稍后重试")
            if msg.get("feedback"):
                st.caption("感谢反馈")
            else:
                feedback_col1, feedback_col2 = st.columns(2)
                with feedback_col1:
                    if st.button("👍 有帮助", key=f"feedback_up_{msg['interaction_id']}"):
                        if submit_feedback(msg["interaction_id"], "positive"):
                            msg["feedback"] = "positive"
                            st.rerun()
                        st.warning("反馈提交失败")
                with feedback_col2:
                    if st.button("👎 没帮助", key=f"feedback_down_{msg['interaction_id']}"):
                        if submit_feedback(msg["interaction_id"], "negative"):
                            msg["feedback"] = "negative"
                            st.rerun()
                        st.warning("反馈提交失败")

# 输入框
if prompt := st.chat_input("请输入您的问题..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 调用API
    with st.chat_message("assistant"):
        status = st.empty()
        timing = st.empty()
        answer_area = st.empty()
        status.info("正在连接问答服务")
        try:
            # 保持一次请求持续读取，页面刷新和反馈不会再发起一次模型问答。
            with requests.post(
                f"{API_BASE}/qa/ask/stream",
                json={"question": prompt, "session_id": st.session_state.current_session},
                stream=True, timeout=(5, 60),
            ) as response:
                if response.status_code == 200:
                    answer, data = "", None
                    for event in read_answer_events(response):
                        payload = event["data"]
                        if event["event"] == "status":
                            status.info(payload.get("message", "正在处理问题"))
                        elif event["event"] == "delta":
                            answer += payload["text"]
                            answer_area.markdown(answer + " ▌")
                            if payload.get("first_token_ms") is not None:
                                timing.caption(f"首字耗时：{payload['first_token_ms'] / 1000:.2f} 秒（后台接收至首段正文）")
                        elif event["event"] == "error":
                            raise ValueError(payload.get("answer", "回答生成失败"))
                        elif event["event"] == "done":
                            data = payload
                            break
                    if data is None:
                        raise ValueError("连接中断，回答未完成，请重新提问")
                    # 以完成消息为准；只有完整结果进入会话和反馈流程。
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    answer_area.markdown(answer)
                    if (data.get("query_details") or {}).get("analysis_type") == "agent":
                        render_agent_report(data.get("query_details"))
                    elif (data.get("query_details") or {}).get("analysis_type") in {"creator_commerce", "shop_live"}:
                        render_commerce_analysis(data.get("query_details"))
                    else:
                        render_sales_query(data.get("query_details"))
                    status.empty()
                    if sources:
                        with st.expander("📚 引用来源"):
                            for src in sources:
                                st.caption(f"**{src['source']}**")
                                st.text(src["content"][:300] + "..." if len(src.get("content", "")) > 300 else src.get("content", ""))
                    
                    # 保存助手消息
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "interaction_id": data.get("interaction_id"),
                        "trace_id": data.get("trace_id"),
                        "first_token_ms": data.get("first_token_ms"),
                        "query_details": data.get("query_details"),
                    })
                else:
                    status.empty()
                    if response.status_code == 404:
                        st.error("后端尚未加载流式问答接口，请重启后端后再试")
                    else:
                        st.error(f"API错误 ({response.status_code}): {response.text}")
                    data = None
            if data is not None:
                st.rerun()
        except requests.exceptions.ConnectionError:
            status.empty()
            st.error("❌ 无法连接到API服务，请确保后端已启动；本次回答未完成")
        except Exception as e:
            status.empty()
            st.error(f"回答未完成：{str(e)}")
