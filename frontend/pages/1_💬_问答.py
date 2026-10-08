import streamlit as st
import requests
import json
import uuid
import streamlit.components.v1 as components

# API基础地址
API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(page_title="智能问答", page_icon="💬")

st.title("💬 智能问答")
st.caption("基于知识库的智能问答，支持多轮对话")

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
        st.session_state.current_session = str(uuid.uuid4())
        st.session_state.sessions = {st.session_state.current_session: "新会话"}
        save_sessions()
    if "current_session" not in st.session_state:
        st.session_state.current_session = list(st.session_state.sessions.keys())[0]
if "messages" not in st.session_state:
    st.session_state.messages = []

# 侧边栏会话管理
with st.sidebar:
    st.subheader("📋 会话管理")
    
    # 显示会话列表
    selected_session = st.selectbox(
        "选择会话",
        options=list(st.session_state.sessions.values()),
        format_func=lambda x: x,
        index=0
    )
    
    # 切换会话
    if selected_session != st.session_state.sessions[st.session_state.current_session]:
        for session_id, session_name in st.session_state.sessions.items():
            if session_name == selected_session:
                st.session_state.current_session = session_id
                save_sessions()
                # 加载会话历史
                try:
                    response = requests.get(f"{API_BASE}/qa/history?session_id={session_id}", timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.messages = data.get("history", [])
                except:
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
    st.caption("提示：系统会记住最近5轮对话")

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 查看引用来源"):
                for src in msg["sources"]:
                    st.caption(f"**{src['source']}** (页码:{src.get('page','-')})")
                    st.text(src["content"][:200] + "..." if len(src.get("content", "")) > 200 else src.get("content", ""))

# 输入框
if prompt := st.chat_input("请输入您的问题..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 调用API
    with st.chat_message("assistant"):
        with st.spinner("🤔 思考中..."):
            try:
                response = requests.post(
                    f"{API_BASE}/qa/ask",
                    json={"question": prompt, "session_id": st.session_state.current_session},
                    timeout=60
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    
                    st.markdown(answer)
                    if sources:
                        with st.expander("📚 引用来源"):
                            for src in sources:
                                st.caption(f"**{src['source']}**")
                                st.text(src["content"][:300] + "..." if len(src.get("content", "")) > 300 else src.get("content", ""))
                    
                    # 保存助手消息
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error(f"API错误 ({response.status_code}): {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("❌ 无法连接到API服务，请确保后端已启动")
            except Exception as e:
                st.error(f"请求失败: {str(e)}")
