"""Ingestion Traces page – browse ingestion trace history with stage waterfall.

Layout:
1. Trace list (reverse-chronological, filtered to trace_type=="ingestion")
2. Detail view: horizontal bar chart showing load/split/transform/embed/upsert
   elapsed times.
"""

from __future__ import annotations

import streamlit as st

from src.observability.dashboard.services.trace_service import TraceService


def render() -> None:
    """Render the Ingestion Traces page."""
    st.header("🔬 Ingestion Traces")

    svc = TraceService()
    traces = svc.list_traces(trace_type="ingestion")

    if not traces:
        st.info("No ingestion traces recorded yet. Run an ingestion first!")
        return

    st.subheader(f"📋 Trace History ({len(traces)})")

    for idx, trace in enumerate(traces):
        trace_id = trace.get("trace_id", "unknown")
        started = trace.get("started_at", "—")
        total_ms = trace.get("elapsed_ms")
        total_label = f"{total_ms:.0f} ms" if total_ms is not None else "—"

        with st.expander(
            f"**{trace_id[:12]}…** — {started} — total: {total_label}",
            expanded=(idx == 0),
        ):
            # ── Stage waterfall chart ──────────────────────────────
            timings = svc.get_stage_timings(trace)

            if timings:
                st.markdown("**Stage Timings**")

                # Build data for a simple horizontal bar chart
                chart_data = {
                    t["stage_name"]: t["elapsed_ms"] for t in timings
                }

                st.bar_chart(chart_data, horizontal=True)

                # Also show a table for exact numbers
                st.table(
                    [
                        {
                            "Stage": t["stage_name"],
                            "Elapsed (ms)": round(t["elapsed_ms"], 2),
                        }
                        for t in timings
                    ]
                )
            else:
                st.caption("No stage timings recorded.")

            # ── Metadata ───────────────────────────────────────────
            meta = trace.get("metadata", {})
            if meta:
                with st.expander("Metadata", expanded=False):
                    st.json(meta)
