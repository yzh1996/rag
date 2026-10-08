"""
聊天相关组件
"""
import streamlit as st
from typing import List, Dict, Any, Optional


def render_chat_message(
    role: str, 
    content: str, 
    sources: Optional[List[Dict[str, Any]]] = None,
    avatar: Optional[str] = None
):
    """
    渲染单条聊天消息
    :param role: "user" 或 "assistant"
    :param content: 消息内容
    :param sources: 引用来源列表
    :param avatar: 头像emoji
    """
    if avatar is None:
        avatar = "👤" if role == "user" else "🤖"
    
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)
        if sources:
            render_sources(sources)


def render_sources(sources: List[Dict[str, Any]], expanded: bool = False):
    """
    渲染引用来源
    """
    with st.expander("📚 引用来源", expanded=expanded):
        for i, src in enumerate(sources, 1):
            source_name = src.get("source", "未知文档")
            page = src.get("page")
            content = src.get("content", "")
            
            # 构建来源标题
            title = f"**{i}. {source_name}**"
            if page:
                title += f" (第{page}页)"
            
            st.markdown(title)
            with st.container():
                st.text(content[:200] + "..." if len(content) > 200 else content)
            if i < len(sources):
                st.divider()


def render_chat_input(
    placeholder: str = "请输入您的问题...",
    key: str = "chat_input",
    on_submit: Optional[callable] = None
) -> Optional[str]:
    """
    渲染聊天输入框
    """
    return st.chat_input(placeholder=placeholder, key=key)


def render_chat_history(
    messages: List[Dict[str, Any]],
    show_sources: bool = True
):
    """
    渲染完整的聊天历史
    """
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        sources = msg.get("sources") if show_sources else None
        render_chat_message(role, content, sources)


def render_typing_indicator():
    """
    渲染"正在输入"指示器
    """
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("思考中..."):
            st.empty()


def render_welcome_message():
    """
    渲染欢迎消息
    """
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown("""
        您好！我是企业知识库助手 🤖
        
        我可以帮您：
        - 📄 检索内部文档并回答问题
        - 🔍 查找相关政策、制度、流程
        - 💡 提供基于知识库的专业建议
        
        请随时向我提问！
        """)


def copy_to_clipboard_button(text: str, button_text: str = "📋 复制"):
    """
    渲染复制到剪贴板的按钮
    """
    # 使用st.code和st.markdown实现复制功能
    st.code(text, language="text")
    st.button(button_text, on_click=lambda: st.write(
        f'<script>navigator.clipboard.writeText(`{text}`)</script>',
        unsafe_allow_html=True
    ))