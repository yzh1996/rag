"""Render actual agent execution steps without empty metric reports."""
import pandas as pd
import streamlit as st


LABELS = {"discover_capabilities": "获取相关能力", "read_business_schema": "读取表与字段说明",
    "query_database": "查询本地数据", "search_knowledge": "检索内部知识", "calculate": "计算对比结果",
    "finish_answer": "整理回答", "model": "模型调用"}


def render_agent_report(details):
    if not details or details.get("analysis_type") != "agent":
        return
    if details.get("status") == "failed":
        st.warning("本次处理未完成，可展开执行过程查看失败位置。")
    elif details.get("status") == "partial":
        st.info("已达到处理预算，结果仅基于已完成的查询，可继续追问。")
    elif details.get("status") == "clarify":
        st.info("需要补充信息，已保存当前确认的内容。")
    steps = details.get("steps", [])
    if not steps:
        st.caption("本次直接回答，未调用工具。")
        return
    with st.expander("查看执行过程与数据依据"):
        for step in steps:
            label = LABELS.get(step["tool"], step["tool"])
            status = {"succeeded": "完成", "failed": "失败", "no_results": "未找到资料"}.get(step["status"], step["status"])
            st.write(f"{step['number']}. {label} · {status}")
            st.caption(f"{step.get('summary', '')} · {step.get('elapsed_ms', 0):.0f} 毫秒")
            if step["status"] == "failed" and step.get("arguments", {}).get("sql"):
                st.code(step["arguments"]["sql"], language="sql")
        for query in details.get("queries", []):
            st.write("数据来源：" + "、".join(query.get("tables", [])))
            if query.get("dataset_label") == "SIMULATED":
                st.caption("本地 MySQL 模拟数据")
            st.code(query.get("sql", ""), language="sql")
            if query.get("rows"):
                st.dataframe(pd.DataFrame(query["rows"], columns=query["columns"]),
                    use_container_width=True, hide_index=True)
            else:
                st.caption("未查到匹配记录，不代表指标为零。")
            if query.get("truncated"):
                st.info("结果达到返回上限，请缩小范围查看其余数据。")
