"""Data Browser page – browse ingested documents, chunks, and images.

Layout:
1. Collection filter (sidebar / top)
2. Document list table (source_path, collection, chunk_count, image_count, processed_at)
3. Chunk detail expander (text, metadata)
4. Image preview (if any)
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.observability.dashboard.services.data_service import DataService


def render() -> None:
    """Render the Data Browser page."""
    st.header("🔍 Data Browser")

    try:
        svc = DataService()
    except Exception as exc:
        st.error(f"Failed to initialise DataService: {exc}")
        return

    # ── Collection filter ──────────────────────────────────────────
    collection = st.text_input(
        "Filter by collection (leave blank for all)",
        value="",
        key="db_collection_filter",
    )
    coll_arg = collection.strip() if collection.strip() else None

    # ── Document list ──────────────────────────────────────────────
    try:
        docs = svc.list_documents(coll_arg)
    except Exception as exc:
        st.error(f"Failed to load documents: {exc}")
        return

    if not docs:
        st.info("No documents found. Ingest some data first!")
        return

    st.subheader(f"📄 Documents ({len(docs)})")

    for idx, doc in enumerate(docs):
        label = f"{doc['source_path']}  |  chunks: {doc['chunk_count']}  |  images: {doc['image_count']}"
        with st.expander(label, expanded=False):
            st.markdown(
                f"- **Source hash:** `{doc['source_hash']}`\n"
                f"- **Collection:** `{doc.get('collection', '—')}`\n"
                f"- **Processed at:** {doc.get('processed_at', '—')}"
            )

            # ── Chunk details ──────────────────────────────────────
            if st.button("Load chunks", key=f"load_chunks_{idx}"):
                chunks = svc.get_chunks(doc["source_hash"])
                if chunks:
                    for cidx, chunk in enumerate(chunks):
                        with st.expander(
                            f"Chunk {cidx + 1}: {chunk['id']}", expanded=False
                        ):
                            st.text_area(
                                "Content",
                                value=chunk.get("text", ""),
                                height=150,
                                disabled=True,
                                key=f"chunk_text_{idx}_{cidx}",
                            )
                            st.json(chunk.get("metadata", {}))
                else:
                    st.caption("No chunks found in ChromaDB for this document.")

            # ── Image preview ──────────────────────────────────────
            images = svc.get_images(doc["source_hash"])
            if images:
                st.markdown(f"**🖼️ Images ({len(images)})**")
                img_cols = st.columns(min(len(images), 4))
                for iidx, img in enumerate(images):
                    with img_cols[iidx % len(img_cols)]:
                        img_path = Path(img.get("file_path", ""))
                        if img_path.exists():
                            st.image(str(img_path), caption=img["image_id"], width=200)
                        else:
                            st.caption(f"{img['image_id']} (file missing)")
