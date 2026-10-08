"""
Streamlit主应用入口
"""
import streamlit as st

st.set_page_config(
    page_title="企业RAG智能问答系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 首页内容
st.title("🤖 企业级私有化RAG智能问答系统")
st.markdown("""
### 欢迎使用企业知识库助手

本系统基于RAG（检索增强生成）技术，可帮助您快速检索企业内部文档并获取精准答案。

#### ✨ 核心功能
- 📁 **知识库管理**：上传PDF、Word、Excel等文档，自动向量化存储
- 💬 **智能问答**：自然语言提问，系统结合知识库生成回答
- 🔍 **来源追溯**：每个回答都标注引用来源，确保可信
- 🧠 **多轮对话**：支持上下文连续对话
- ⚙️ **灵活配置**：支持本地Ollama模型或云端API

#### 🚀 快速开始
1. 进入「知识库管理」上传您的文档
2. 进入「智能问答」开始提问
3. 在「系统设置」中配置模型参数

请从侧边栏选择功能模块。
""")

# 显示系统状态
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📄 向量库文档数", "—")
with col2:
    st.metric("💬 今日问答次数", "—")
with col3:
    st.metric("🟢 系统状态", "在线")

st.caption(f"© xxxx 企业级RAG系统 vx.0")