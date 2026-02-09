"""Query Traces page – browse query trace history with stage waterfall.

Layout:
1. Optional keyword search filter
2. Trace list (reverse-chronological, filtered to trace_type=="query")
3. Detail view: stage waterfall + Dense vs Sparse comparison + Rerank delta
4. Per-trace Ragas evaluation button (LLM-as-Judge scoring)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import streamlit as st

from src.observability.dashboard.services.trace_service import TraceService

logger = logging.getLogger(__name__)


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
        meta = trace.get("metadata", {})
        query_text = meta.get("query", "")
        source = meta.get("source", "unknown")

        # ── Expander title: show query text ────────────────────
        query_preview = (
            query_text[:40] + "…" if len(query_text) > 40 else query_text
        ) if query_text else "—"
        expander_title = (
            f"🔍 \"{query_preview}\"  ·  {total_label}  ·  {started[:19]}"
        )

        with st.expander(expander_title, expanded=(idx == 0)):
            # ── 1. Query overview ──────────────────────────────
            st.markdown("#### 💬 Query")
            col_q, col_meta = st.columns([3, 1])
            with col_q:
                st.markdown(f"> {query_text}")
            with col_meta:
                source_emoji = "🤖" if source == "mcp" else "📡"
                st.markdown(f"**Source:** {source_emoji} `{source}`")
                st.markdown(f"**Top-K:** `{meta.get('top_k', '—')}`")
                st.markdown(f"**Collection:** `{meta.get('collection', '—')}`")

            st.divider()

            # ── 2. Retrieval results summary ───────────────────
            timings = svc.get_stage_timings(trace)
            dense = _find_stage(timings, "dense_retrieval")
            sparse = _find_stage(timings, "sparse_retrieval")
            rerank = _find_stage(timings, "rerank")

            dense_count = dense["data"].get("result_count", 0) if dense and dense.get("data") else 0
            sparse_count = sparse["data"].get("result_count", 0) if sparse and sparse.get("data") else 0

            st.markdown("#### 📊 Retrieval Results")
            rc1, rc2, rc3, rc4 = st.columns(4)
            with rc1:
                st.metric("Dense Hits", dense_count)
            with rc2:
                st.metric("Sparse Hits", sparse_count)
            with rc3:
                st.metric("Total Fused", dense_count + sparse_count)
            with rc4:
                st.metric("Total Time", total_label)

            st.divider()

            # ── 3. Stage waterfall chart ───────────────────────
            if timings:
                st.markdown("#### ⏱️ Stage Timings")

                chart_data = {t["stage_name"]: t["elapsed_ms"] for t in timings}
                st.bar_chart(chart_data, horizontal=True)

                st.table(
                    [
                        {
                            "": i,
                            "Stage": t["stage_name"],
                            "Elapsed (ms)": round(t["elapsed_ms"], 2),
                        }
                        for i, t in enumerate(timings)
                    ]
                )

            # ── 4. Dense vs Sparse detail ──────────────────────
            if dense or sparse:
                st.markdown("#### 🔬 Dense vs Sparse Detail")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Dense Retrieval",
                        f"{dense['elapsed_ms']:.1f} ms" if dense else "—",
                    )
                    if dense and dense.get("data"):
                        st.json(dense["data"])
                with col2:
                    st.metric(
                        "Sparse Retrieval",
                        f"{sparse['elapsed_ms']:.1f} ms" if sparse else "—",
                    )
                    if sparse and sparse.get("data"):
                        st.json(sparse["data"])

            # ── 5. Rerank details ──────────────────────────────
            if rerank:
                st.markdown("#### 🔄 Reranking")
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

            # ── 6. Raw metadata ────────────────────────────────
            if meta:
                with st.expander("🗂️ Raw Metadata", expanded=False):
                    st.json(meta)

            # ── 7. Ragas Evaluate button ───────────────────────
            _render_evaluate_button(trace, idx)


def _render_evaluate_button(trace: Dict[str, Any], idx: int) -> None:
    """Render a Ragas evaluate button for a single query trace.

    Re-runs retrieval for the stored query and evaluates with
    RagasEvaluator (LLM-as-Judge).  Only works when query text
    is available in trace metadata.
    """
    meta = trace.get("metadata", {})
    query = meta.get("query", "")
    if not query:
        return

    st.divider()
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        clicked = st.button(
            "📏 Ragas Evaluate",
            key=f"eval_trace_{idx}",
            help="Re-run this query and score with Ragas (LLM-as-Judge)",
        )
    with col_info:
        st.caption(
            "Uses Ragas to score faithfulness, answer relevancy, "
            "and context precision. Calls LLM — may take a few seconds."
        )

    # Show previous result from session state
    result_key = f"eval_result_{idx}"
    if result_key in st.session_state and not clicked:
        _display_eval_metrics(st.session_state[result_key])

    if clicked:
        with st.spinner("Running Ragas evaluation…"):
            result = _evaluate_single_trace(query, meta)
        st.session_state[result_key] = result
        _display_eval_metrics(result)


def _evaluate_single_trace(
    query: str,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Re-run retrieval and evaluate a single query with Ragas.

    Returns dict with 'metrics' (score dict) or 'error' (str).
    """
    try:
        from src.core.settings import load_settings
        from src.libs.evaluator.evaluator_factory import EvaluatorFactory

        settings = load_settings()

        # Create Ragas evaluator
        eval_settings = settings.evaluation
        override = type(eval_settings)(
            enabled=True,
            provider="ragas",
            metrics=eval_settings.metrics if hasattr(eval_settings, "metrics") else [],
        )
        evaluator = EvaluatorFactory.create(override)

        # Re-run retrieval
        collection = meta.get("collection", "default")
        top_k = meta.get("top_k", 10)
        chunks = _retrieve_chunks(settings, query, top_k, collection)

        if not chunks:
            return {"error": "No chunks retrieved — is data indexed?"}

        # Build a simple answer from top chunks
        texts = []
        for c in chunks:
            if hasattr(c, "text"):
                texts.append(c.text)
            elif isinstance(c, dict):
                texts.append(c.get("text", str(c)))
            else:
                texts.append(str(c))
        answer = " ".join(texts[:5])

        # Evaluate
        metrics = evaluator.evaluate(
            query=query,
            retrieved_chunks=chunks,
            generated_answer=answer,
        )
        return {"metrics": metrics}

    except ImportError as exc:
        return {"error": f"Ragas not installed: {exc}"}
    except Exception as exc:
        logger.exception("Ragas evaluation failed")
        return {"error": str(exc)}


def _retrieve_chunks(
    settings: Any,
    query: str,
    top_k: int,
    collection: str,
) -> list:
    """Re-run HybridSearch to retrieve chunks for evaluation."""
    try:
        from src.core.query_engine.hybrid_search import create_hybrid_search
        from src.core.query_engine.query_processor import QueryProcessor
        from src.core.query_engine.dense_retriever import create_dense_retriever
        from src.core.query_engine.sparse_retriever import create_sparse_retriever
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        vector_store = VectorStoreFactory.create(
            settings, collection_name=collection,
        )
        embedding_client = EmbeddingFactory.create(settings)
        dense_retriever = create_dense_retriever(
            settings=settings,
            embedding_client=embedding_client,
            vector_store=vector_store,
        )
        bm25_indexer = BM25Indexer(index_dir=f"data/db/bm25/{collection}")
        sparse_retriever = create_sparse_retriever(
            settings=settings,
            bm25_indexer=bm25_indexer,
            vector_store=vector_store,
        )
        sparse_retriever.default_collection = collection
        query_processor = QueryProcessor()
        hybrid_search = create_hybrid_search(
            settings=settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
        )

        results = hybrid_search.search(query=query, top_k=top_k)
        return results if isinstance(results, list) else results.results
    except Exception as exc:
        logger.warning("Retrieval for evaluation failed: %s", exc)
        return []


def _display_eval_metrics(result: Dict[str, Any]) -> None:
    """Display evaluation result (metrics or error)."""
    if "error" in result:
        st.error(f"❌ Evaluation failed: {result['error']}")
        return

    metrics = result.get("metrics", {})
    if not metrics:
        st.warning("No metrics returned.")
        return

    st.markdown("**📏 Ragas Scores**")
    cols = st.columns(min(len(metrics), 4))
    for i, (name, value) in enumerate(sorted(metrics.items())):
        with cols[i % len(cols)]:
            st.metric(
                label=name.replace("_", " ").title(),
                value=f"{value:.4f}",
            )


def _find_stage(timings, name):
    """Find a stage dict by name, or None."""
    for t in timings:
        if t["stage_name"] == name:
            return t
    return None
