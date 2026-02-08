"""Query Traces page – browse query trace history with stage waterfall.

Layout:
1. Optional keyword search filter
2. Trace list (reverse-chronological, filtered to trace_type=="query")
3. Detail view: stage waterfall + Dense vs Sparse comparison + Rerank delta
"""

from __future__ import annotations

import streamlit as st

from src.observability.dashboard.services.trace_service import TraceService


def render() -> None:
    """Render the Query Traces page."""
    st.header("🔎 Query Traces")

    svc = TraceService()
    traces = svc.list_traces(trace_type="query")

    if not traces:
        st.info("No query traces recorded yet. Run a query first!")
        return

    # ── Keyword filter ─────────────────────────────────────────────
    keyword = st.text_input(
        "Search by query keyword",
        value="",
        key="qt_keyword",
    )
    if keyword.strip():
        kw = keyword.strip().lower()
        traces = [
            t
            for t in traces
            if kw in str(t.get("metadata", {})).lower()
            or kw in str(t.get("stages", [])).lower()
        ]

    st.subheader(f"📋 Query History ({len(traces)})")

    for idx, trace in enumerate(traces):
        trace_id = trace.get("trace_id", "unknown")
        started = trace.get("started_at", "—")
        total_ms = trace.get("elapsed_ms")
        total_label = f"{total_ms:.0f} ms" if total_ms is not None else "—"

        with st.expander(
            f"**{trace_id[:12]}…** — {started} — total: {total_label}",
            expanded=(idx == 0),
        ):
            timings = svc.get_stage_timings(trace)

            # ── Stage waterfall chart ──────────────────────────────
            if timings:
                st.markdown("**Stage Timings**")

                chart_data = {t["stage_name"]: t["elapsed_ms"] for t in timings}
                st.bar_chart(chart_data, horizontal=True)

                st.table(
                    [
                        {
                            "Stage": t["stage_name"],
                            "Elapsed (ms)": round(t["elapsed_ms"], 2),
                        }
                        for t in timings
                    ]
                )

            # ── Dense vs Sparse comparison ─────────────────────────
            dense = _find_stage(timings, "dense_retrieval")
            sparse = _find_stage(timings, "sparse_retrieval")

            if dense or sparse:
                st.markdown("**Dense vs Sparse Retrieval**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Dense Retrieval",
                        f"{dense['elapsed_ms']:.1f} ms" if dense else "—",
                    )
                    if dense and dense["data"]:
                        st.json(dense["data"])
                with col2:
                    st.metric(
                        "Sparse Retrieval",
                        f"{sparse['elapsed_ms']:.1f} ms" if sparse else "—",
                    )
                    if sparse and sparse["data"]:
                        st.json(sparse["data"])

            # ── Rerank details ─────────────────────────────────────
            rerank = _find_stage(timings, "rerank")
            if rerank:
                st.markdown("**Reranking**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Elapsed", f"{rerank['elapsed_ms']:.1f} ms")
                with col2:
                    st.metric(
                        "Input count",
                        rerank["data"].get("input_count", "—"),
                    )
                with col3:
                    st.metric(
                        "Output count",
                        rerank["data"].get("output_count", "—"),
                    )

            # ── Metadata ───────────────────────────────────────────
            meta = trace.get("metadata", {})
            if meta:
                with st.expander("Metadata", expanded=False):
                    st.json(meta)


def _find_stage(timings, name):
    """Find a stage dict by name, or None."""
    for t in timings:
        if t["stage_name"] == name:
            return t
    return None
