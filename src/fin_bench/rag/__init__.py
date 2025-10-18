"""
RAG (Retrieval-Augmented Generation) module for FinBench.

This module provides a complete RAG pipeline for efficient context retrieval
when working with long financial documents.

Components:
- embeddings: Embedding model wrapper
- chunking: Text chunking strategies
- vector_store: Vector storage (FAISS)
- retriever: Retrieval and reranking
- cache: Index caching
- pipeline: Main RAG orchestrator
"""

from .rag_config import RAGConfig
from .pipeline import RAGPipeline

__all__ = [
    "RAGConfig",
    "RAGPipeline",
]
