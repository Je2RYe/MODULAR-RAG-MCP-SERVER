"""DataService – read-only facade for browsing ingested data.

Wraps ``DocumentManager``, ``ChromaStore``, and ``ImageStorage`` to
provide the data the Data Browser page needs, without coupling the
UI to storage internals.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DataService:
    """Provide read-only access to ingested documents, chunks, and images.

    Lazily instantiates the heavy storage objects on first call so that
    importing the module alone has zero cost.
    """

    def __init__(self) -> None:
        self._manager: Any = None
        self._chroma: Any = None
        self._images: Any = None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_stores(self) -> None:
        """Create storage objects on first use."""
        if self._manager is not None:
            return

        from src.core.settings import load_settings
        from src.ingestion.document_manager import DocumentManager
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.ingestion.storage.image_storage import ImageStorage
        from src.libs.loader.file_integrity import SQLiteIntegrityChecker
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        settings = load_settings("config/settings.yaml")

        chroma = VectorStoreFactory.create(settings)
        bm25 = BM25Indexer(
            index_dir=str(
                Path(settings.get("data_dir", "data/db/bm25"))
            )
        )
        images = ImageStorage()
        integrity = SQLiteIntegrityChecker(
            db_path=str(
                Path(settings.get("data_dir", "data/db"))
                / "file_integrity.db"
            )
        )

        self._chroma = chroma
        self._images = images
        self._manager = DocumentManager(chroma, bm25, images, integrity)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_documents(
        self, collection: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return ingested documents as plain dicts (UI-friendly).

        Each dict has keys: source_path, source_hash, collection,
        chunk_count, image_count, processed_at.
        """
        self._ensure_stores()
        from dataclasses import asdict

        docs = self._manager.list_documents(collection)
        return [asdict(d) for d in docs]

    def get_document_detail(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Return document detail as a plain dict, or None."""
        self._ensure_stores()
        from dataclasses import asdict

        detail = self._manager.get_document_detail(doc_id)
        if detail is None:
            return None
        return asdict(detail)

    def get_chunks(self, source_hash: str) -> List[Dict[str, Any]]:
        """Return chunk records from ChromaDB matching *source_hash*.

        Each dict has keys: id, text, metadata.
        """
        self._ensure_stores()
        try:
            results = self._chroma.collection.get(
                where={"source_hash": source_hash},
                include=["documents", "metadatas"],
            )
            chunks: List[Dict[str, Any]] = []
            ids = results.get("ids", [])
            docs = results.get("documents", [])
            metas = results.get("metadatas", [])
            for i, cid in enumerate(ids):
                chunks.append(
                    {
                        "id": cid,
                        "text": docs[i] if docs else "",
                        "metadata": metas[i] if metas else {},
                    }
                )
            return chunks
        except Exception as exc:
            logger.warning("Failed to get chunks for %s: %s", source_hash, exc)
            return []

    def get_images(self, source_hash: str) -> List[Dict[str, Any]]:
        """Return image records for a document."""
        self._ensure_stores()
        try:
            return self._images.list_images(doc_hash=source_hash)
        except Exception as exc:
            logger.warning("Failed to get images for %s: %s", source_hash, exc)
            return []

    def get_collection_stats(
        self, collection: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return aggregate stats as a plain dict."""
        self._ensure_stores()
        from dataclasses import asdict

        stats = self._manager.get_collection_stats(collection)
        return asdict(stats)
