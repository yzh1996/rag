import streamlit as st
import requests
import time

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(page_title="知识库管理", page_icon="📁")

st.title("📁 知识库管理")

# 标签页
tab1, tab2, tab3 = st.tabs(["📤 上传文档", "📊 知识库统计", "⚡ 批量操作"])

with tab1:
    st.subheader("上传文档")
    
    # 文件上传
    uploaded_files = st.file_uploader(
        "选择文件（支持PDF、Word、Excel、TXT、Markdown）",
        type=["pdf", "docx", "doc", "txt", "md", "xlsx", "xls"],
        accept_multiple_files=True
    )
    
    category = st.selectbox(
        "文档分类",
        ["general", "技术文档", "管理制度", "产品手册", "培训材料", "其他"]
    )
    
    if st.button("🚀 开始上传并处理", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("请先选择文件")
        else:
            progress_bar = st.progress(0, text="准备上传...")
            status_text = st.empty()
            
            success_count = 0
            fail_count = 0
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"正在上传: {file.name} ({i+1}/{len(uploaded_files)})")
                progress_bar.progress((i) / len(uploaded_files))
                
                try:
                    files = {"file": (file.name, file.getvalue())}
                    data = {"category": category}
                    response = requests.post(
                        f"{API_BASE}/upload/file",
                        files=files,
                        data=data,
                        timeout=120
                    )
                    if response.status_code == 200:
                        success_count += 1
                        st.toast(f"✅ {file.name} 上传成功")
                    else:
                        fail_count += 1
                        st.toast(f"❌ {file.name} 上传失败")
                except Exception as e:
                    fail_count += 1
                    st.toast(f"❌ {file.name} 错误: {str(e)}")
                
                time.sleep(0.5)
            
            progress_bar.progress(1.0, text="上传完成！")
            status_text.text(f"上传完成！成功: {success_count}, 失败: {fail_count}")
            
            if success_count > 0:
                st.success(f"✅ 成功上传 {success_count} 个文件，正在后台处理...")

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