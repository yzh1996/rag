"""
侧边栏组件
"""
import streamlit as st
from typing import Optional, Callable


def render_sidebar(
    title: str = "导航菜单",
    on_clear_history: Optional[Callable] = None
) -> str:
    """
    渲染通用侧边栏
    :param title: 侧边栏标题
    :param on_clear_history: 清除历史的回调函数
    :return: 当前选中的页面
    """
    with st.sidebar:
        st.title("🤖 RAG系统")
        st.divider()
        
        # 导航菜单
        st.subheader(title)
        page = st.radio(
            "选择功能",
            ["💬 智能问答", "📁 知识库管理", "⚙️ 系统设置", "📊 仪表盘"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # 会话管理
        st.subheader("会话管理")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 新会话", use_container_width=True):
                if "messages" in st.session_state:
                    st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("🗑️ 清除历史", use_container_width=True):
                if on_clear_history:
                    on_clear_history()
                if "messages" in st.session_state:
                    st.session_state.messages = []
                st.rerun()
        
        # 系统状态
        st.divider()
        st.subheader("系统状态")
        
        # 显示API连接状态
        if st.session_state.get("api_connected", False):
            st.success("🟢 API在线")
        else:
            st.error("🔴 API离线")
        
        # 显示会话信息
        if "session_id" in st.session_state:
            st.caption(f"会话ID: {st.session_state.session_id[:8]}...")
        
        st.divider()
        st.caption("© 2024 企业级RAG系统")
        
        return page


def render_settings_sidebar():
    """
    渲染设置页面的侧边栏
    """
    with st.sidebar:
        st.title("⚙️ 设置")
        st.divider()
        
        st.subheader("检索设置")
        top_k = st.slider("返回文档数量", 1, 20, 5)
        use_rerank = st.checkbox("启用重排序", value=False)
        
        st.subheader("模型设置")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
        max_tokens = st.number_input("最大输出长度", 256, 4096, 2048)
        
        return {
            "top_k": top_k,
            "use_rerank": use_rerank,
            "temperature": temperature,
            "max_tokens": max_tokens
        }