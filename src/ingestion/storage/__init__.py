"""
Storage Module.

This package contains storage components:
- Vector upserter
- BM25 indexer
- Image storage
"""

from src.ingestion.storage.bm25_indexer import BM25Indexer

__all__ = ["BM25Indexer"]
