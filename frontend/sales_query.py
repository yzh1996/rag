"""展示本轮和历史问答中的数据库查询结果。"""
import pandas as pd
import streamlit as st


def render_sales_query(details):
    if not details:
        return
    if details.get("period_start"):
        st.caption(f"数据快照：{details['period_start']} 至 {details['period_end']} · 单位：件")
        st.caption(f"数据更新时间：{details.get('updated_at', '未知')}（UTC）")
    if details.get("status") == "succeeded":
        st.dataframe(pd.DataFrame(details.get("rows", []), columns=details.get("columns", [])),
            use_container_width=True, hide_index=True)
        if details.get("truncated"):
            st.info("结果已达到返回上限，请缩小查询范围查看其余记录。")
    with st.expander("查看数据库查询详情"):
        retrieval = details.get("schema_retrieval") or {}
        if retrieval.get("tables"):
            st.write("本次使用的表：" + "、".join(retrieval["tables"]))
            for hit in retrieval.get("hits", []):
                score = f" · 相似度 {hit['similarity']:.3f}" if hit.get("similarity") is not None else ""
                st.caption(f"{hit['title']} · {hit.get('reason', '向量匹配')}{score}")
        if retrieval.get("status") == "validated":
            st.caption("所选表的字段、类型和主键已与当前数据库核对。")
        if details.get("sql"):
            st.code(details["sql"], language="sql")
        st.caption(f"查询状态：{ {'succeeded': '成功', 'failed': '失败', 'refused': '未执行'}.get(details.get('status'), '未完成') }"
            f" · 耗时：{details.get('elapsed_ms', 0):.0f} 毫秒")
        if details.get("error"):
            st.write(details["error"])
