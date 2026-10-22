import streamlit as st
from frontend.api_client import client as requests, API_BASE, render_auth
from frontend.management import render_management


st.set_page_config(page_title="知识库管理", page_icon="📁")
render_auth()

st.title("📁 知识库管理")

# 标签页
tab1, tab2, tab3, tab4 = st.tabs(["📤 上传文档", "📊 知识库统计", "⚡ 批量操作", "版本与发布"])

with tab4:
    render_management()

with tab1:
    from frontend.upload_flow import render_upload_flow
    render_upload_flow()

with tab2:
    st.subheader("知识库统计")
    
    if st.button("🔄 刷新统计", use_container_width=True):
        try:
            response = requests.get(f"{API_BASE}/knowledge/stats", timeout=10)
            if response.status_code == 200:
                data = response.json()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📄 文档总数", data.get("total_documents", 0))
                with col2:
                    st.metric("🗄️ 向量库类型", data.get("vector_store_type", "N/A"))
                with col3:
                    st.metric("💾 存储路径", data.get("persist_directory", "N/A"))
            else:
                st.error("获取统计信息失败")
        except Exception as e:
            st.error(f"连接API失败: {str(e)}")

with tab3:
    st.subheader("批量操作")
    st.warning("⚠️ 以下操作不可逆，请谨慎操作！")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 清空知识库", type="secondary", use_container_width=True):
            st.session_state.confirm_clear = True
    
    if st.session_state.get("confirm_clear", False):
        st.error("确认清空知识库？这将删除所有向量数据！")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("✅ 确认清空", use_container_width=True):
                try:
                    response = requests.post(f"{API_BASE}/knowledge/clear", timeout=30)
                    if response.status_code == 200:
                        st.success("知识库已清空")
                        st.session_state.confirm_clear = False
                    else:
                        st.error("清空失败")
                except Exception as e:
                    st.error(f"操作失败: {str(e)}")
        with col_b:
            if st.button("❌ 取消", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()
