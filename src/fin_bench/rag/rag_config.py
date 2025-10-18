"""
RAG configuration for FinBench.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline.

    This controls all aspects of the retrieval-augmented generation pipeline,
    from embedding models to chunking strategies to retrieval parameters.
    """

    # Core settings
    enabled: bool = False
    """Whether RAG is enabled. If False, falls back to full context."""

    # Embedding model
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    """HuggingFace embedding model name. Options:
    - BAAI/bge-large-en-v1.5: Best quality, 1024 dims (RECOMMENDED)
    - BAAI/bge-small-en-v1.5: Fast, 384 dims
    - sentence-transformers/all-mpnet-base-v2: Good balance, 768 dims
    """

    embedding_batch_size: int = 32
    """Batch size for embedding generation (higher = faster but more memory)."""

    # Chunking strategy
    chunk_size: int = 512
    """Target chunk size in tokens. Recommended: 256-512 for most cases."""

    chunk_overlap: int = 50
    """Overlap between chunks in tokens. Recommended: 10-20% of chunk_size."""

    use_semantic_chunking: bool = False
    """Use semantic boundaries (sentences/paragraphs) instead of fixed size.
    More accurate but slower."""

    # Retrieval settings
    top_k: int = 20
    """Number of initial chunks to retrieve before reranking."""

    top_k_rerank: int = 5
    """Final number of chunks after reranking. These go to the LLM."""

    similarity_metric: str = "cosine"
    """Similarity metric for vector search. Options: cosine, l2, inner_product."""

    # Reranking
    use_reranker: bool = True
    """Whether to use a cross-encoder reranker for better quality."""

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    """Cross-encoder model for reranking. Options:
    - cross-encoder/ms-marco-MiniLM-L-6-v2: Fast, good quality (RECOMMENDED)
    - cross-encoder/ms-marco-MiniLM-L-12-v2: Better quality, slower
    """

    # Hybrid search (optional)
    use_hybrid_search: bool = False
    """Combine dense (embedding) + sparse (BM25) retrieval."""

    hybrid_alpha: float = 0.7
    """Weight for dense retrieval in hybrid mode. (1-alpha) = BM25 weight."""

    # Vector store
    vector_store_type: str = "faiss"
    """Vector store backend. Options: faiss, chroma."""

    faiss_index_type: str = "IndexFlatIP"
    """FAISS index type. Options:
    - IndexFlatIP: Exact search, inner product (RECOMMENDED for accuracy)
    - IndexFlatL2: Exact search, L2 distance
    - IndexIVFFlat: Approximate, faster for large datasets
    """

    use_gpu: bool = False
    """Use GPU for FAISS operations (requires faiss-gpu)."""

    # Caching
    cache_dir: str = ".rag_cache"
    """Directory for cached indices and embeddings."""

    cache_enabled: bool = True
    """Enable caching of vector indices."""

    # Debug and logging
    debug: bool = False
    """Enable debug logging (shows chunk details, retrieval scores, etc.)."""

    print_chunks: bool = False
    """Print all chunks when indexing (for debugging)."""

    print_retrieval: bool = False
    """Print retrieval results and scores (for debugging)."""

    log_retrieval_stats: bool = True
    """Log retrieval statistics (timing, number of chunks, etc.)."""

    # Advanced options
    normalize_embeddings: bool = True
    """Normalize embeddings to unit length (recommended for cosine similarity)."""

    add_metadata_to_chunks: bool = True
    """Include chunk position and context info in metadata."""

    context_window_expansion: int = 0
    """Expand retrieved chunks by N neighboring chunks for more context."""

    def __post_init__(self):
        """Validate configuration."""
        if self.enabled:
            if self.chunk_size <= 0:
                raise ValueError("chunk_size must be > 0")
            if self.chunk_overlap >= self.chunk_size:
                raise ValueError("chunk_overlap must be < chunk_size")
            if self.top_k_rerank > self.top_k:
                raise ValueError("top_k_rerank must be <= top_k")
            if not 0 <= self.hybrid_alpha <= 1:
                raise ValueError("hybrid_alpha must be between 0 and 1")

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            k: v for k, v in self.__dict__.items()
        }

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create from dictionary."""
        return cls(**config_dict)

    def get_cache_key(self, context: str) -> str:
        """Generate cache key for a context.

        Uses hash of context + relevant config parameters.
        """
        import hashlib

        # Include config params that affect indexing
        config_str = f"{self.embedding_model}_{self.chunk_size}_{self.chunk_overlap}"
        content_str = f"{config_str}_{context}"

        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
