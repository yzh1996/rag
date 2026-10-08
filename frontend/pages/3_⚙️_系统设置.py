import streamlit as st
import requests

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(page_title="系统设置", page_icon="⚙️")

st.title("⚙️ 系统设置")

# 获取当前配置
try:
    response = requests.get(f"{API_BASE}/system/config", timeout=5)
    if response.status_code == 200:
        config = response.json()
    else:
        config = {}
except:
    config = {}

st.subheader("当前系统配置")
st.json(config)

st.divider()
st.subheader("模型配置")
st.info("配置修改功能需要在后端重启后生效，当前为只读展示。")

col1, col2 = st.columns(2)
with col1:
    st.selectbox(
        "LLM提供商",
        ["ollama", "openai", "siliconflow"],
        index=0,
        disabled=True
    )
with col2:
    st.text_input("模型名称", value=config.get("llm_model", ""), disabled=True)

st.text_input("API地址", value="http://localhost:11434", disabled=True)

st.divider()
st.subheader("检索配置")
col1, col2 = st.columns(2)
with col1:
    st.number_input("检索返回数量 (Top-K)", value=config.get("search_top_k", 5), min_value=1, max_value=20, disabled=True)
with col2:
    st.number_input("文档分块大小", value=config.get("chunk_size", 500), min_value=100, max_value=2000, disabled=True)

st.caption("💡 提示：修改配置请编辑项目根目录下的 `.env` 文件，然后重启服务。")