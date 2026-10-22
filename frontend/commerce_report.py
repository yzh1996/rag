"""Render structured commerce-analysis evidence beside the chat answer."""
import pandas as pd
import streamlit as st


def render_commerce_analysis(details):
    if details and details.get("analysis_type") == "shop_live":
        _render_shop_live(details)
        return
    if not details or details.get("analysis_type") != "creator_commerce":
        return
    st.caption(f"演示数据 · {details.get('period_start', '')} 至 {details.get('period_end', '')} · {details.get('period_assumption', '')}")
    specs = details.get("tools_discovered", [])
    labels = {
        "analyze_anchor_live": "达人 · 直播分析",
        "analyze_anchor_video": "达人 · 视频分析",
        "analyze_anchor_products": "达人 · 商品与品类",
        "compare_anchor_category": "同类达人对比",
        "search_selection_knowledge": "选品知识库",
    }
    if specs:
        st.caption("本次分析：" + "、".join(labels.get(item, item) for item in specs if item in labels))
    results = details.get("results") or {}
    metrics = details.get("calculated_metrics") or {}
    live = results.get("analyze_anchor_live") or {}
    video = results.get("analyze_anchor_video") or {}
    first, second, third = st.columns(3)
    first.metric("直播销量", f"{live.get('sales_quantity', '无数据'):,}" if isinstance(live.get("sales_quantity"), int) else "无数据", "件")
    second.metric("视频销量", f"{video.get('sales_quantity', '无数据'):,}" if isinstance(video.get("sales_quantity"), int) else "无数据", "件")
    total_sales = metrics.get("total_sales_quantity")
    third.metric("两渠道合计", f"{total_sales:,}" if isinstance(total_sales, int) else "渠道数据不完整", "件")
    products = results.get("analyze_anchor_products") or {}
    categories = products.get("categories") or []
    if categories:
        st.dataframe(pd.DataFrame(categories), use_container_width=True, hide_index=True)
    comparison = results.get("compare_anchor_category") or {}
    if comparison:
        if comparison.get("status") == "error":
            st.info(comparison.get("message", "同类对比数据暂不可用"))
        else:
            st.write({"同类账号样本数": comparison.get("peer_count"),
                "同类模拟账号销量中位数": comparison.get("peer_median_sales_quantity"),
                "样本提示": comparison.get("sample_size_warning")})
    knowledge = results.get("search_selection_knowledge") or {}
    items = knowledge.get("items") or []
    if items:
        with st.expander(f"查看选品知识依据（{len(items)}条）"):
            for item in items:
                st.markdown(f"**{item.get('source', '内部知识库')}**")
                st.caption(item.get("content", "")[:1200])
    with st.expander("查看分析过程和数据来源"):
        st.write({"已调用工具": details.get("tools_used", []),
            "程序计算指标": details.get("calculated_metrics", {}),
            "知识来源": details.get("knowledge_sources", []),
            "数据标记": details.get("dataset_label", "unknown")})
        limitations = details.get("limitations") or []
        if limitations:
            st.warning("；".join(limitations))


def _render_shop_live(details):
    st.caption(f"本机 MySQL · 模拟数据 · {details.get('period_start', '')} 至 {details.get('period_end', '')}")
    if details.get("status") != "succeeded":
        return
    summary = details.get("summary", {})
    changes = details.get("change_percent", {})
    def delta(key):
        value = changes.get(key)
        return f"较上一场 {value:+.2f}%" if value is not None else None
    first, second, third = st.columns(3)
    first.metric("直播销售额（元）", f"{summary.get('gmv', 0):,.2f}", delta("gmv"))
    second.metric("成交件数", f"{summary.get('sales_quantity', 0):,}", delta("sales_quantity"))
    third.metric("累计观众数", f"{summary.get('viewers', 0):,}", delta("viewers"))
    for session in details.get("sessions", []):
        st.caption(f"直播ID：{session['room_id']} · 开播时间：{session['live_create_time']} · 主播：{session['anchor_name']}")
        products = session.get("products", [])
        if products:
            data = pd.DataFrame(products).rename(columns={"product_id": "商品ID", "product_name": "商品",
                "sales_quantity": "成交件数", "min_price": "最低价（元）", "estimated_gmv": "按最低价估算金额（元）"})
            st.dataframe(data[["商品ID", "商品", "成交件数", "最低价（元）", "按最低价估算金额（元）"]],
                         use_container_width=True, hide_index=True)
    previous = details.get("previous_session")
    if previous:
        st.caption(f"对比上一场：{previous['room_id']} · {previous['live_create_time']}")
    with st.expander("查看分析过程和数据来源"):
        st.write("本次使用的业务表：" + "、".join(details.get("tables", [])))
        st.write(details.get("period_assumption", ""))
        st.json({"summary": summary, "sessions": details.get("sessions", []), "change_percent": changes})
        for limitation in details.get("limitations", []):
            st.caption(limitation)
