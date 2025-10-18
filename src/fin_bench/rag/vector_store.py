"""
Vector store implementations for RAG pipeline.

Provides FAISS-based vector storage for efficient similarity search.
"""

import logging
import pickle
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

from .chunking import Chunk


logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from vector search."""

    chunk: Chunk
    """The retrieved chunk."""

    score: float
    """Similarity score."""

    rank: int
    """Rank in results (0-indexed)."""

    def __repr__(self):
        return f"SearchResult(rank={self.rank}, score={self.score:.4f}, chunk={self.chunk})"


class VectorStore:
    """Base class for vector stores."""

    def add(self, chunks: List[Chunk], embeddings: np.ndarray):
        """Add chunks with their embeddings to the store."""
        raise NotImplementedError

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        """Search for similar chunks."""
        raise NotImplementedError

    def save(self, path: Path):
        """Save vector store to disk."""
        raise NotImplementedError

    def load(self, path: Path):
        """Load vector store from disk."""
        raise NotImplementedError


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store for efficient similarity search.

    Uses FAISS (Facebook AI Similarity Search) for fast approximate
    nearest neighbor search.
    """

    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "IndexFlatIP",
        normalize: bool = True,
        use_gpu: bool = False,
        debug: bool = False
    ):
        """Initialize FAISS vector store.

        Args:
            embedding_dim: Dimension of embeddings
            index_type: FAISS index type (IndexFlatIP, IndexFlatL2, IndexIVFFlat)
            normalize: Normalize vectors before adding (recommended for cosine similarity)
            use_gpu: Use GPU acceleration (requires faiss-gpu)
            debug: Enable debug logging
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.normalize = normalize
        self.use_gpu = use_gpu
        self.debug = debug

        # Import FAISS
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError(
                "faiss not installed. Install with: "
                "pip install faiss-cpu (or faiss-gpu for GPU support)"
            )

        # Create FAISS index
        self.index = self._create_index()

        # Store chunks (FAISS only stores vectors, not metadata)
        self.chunks: List[Chunk] = []

        if self.debug:
            logger.debug(
                f"Created FAISS index: {index_type} "
                f"(dim={embedding_dim}, normalize={normalize}, gpu={use_gpu})"
            )

    def _create_index(self):
        """Create FAISS index based on type."""
        if self.index_type == "IndexFlatIP":
            # Inner product (for normalized vectors, this is cosine similarity)
            index = self.faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "IndexFlatL2":
            # L2 distance
            index = self.faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == "IndexIVFFlat":
            # Inverted file index (approximate, faster for large datasets)
            # Use 100 clusters by default
            quantizer = self.faiss.IndexFlatIP(self.embedding_dim)
            index = self.faiss.IndexIVFFlat(
                quantizer,
                self.embedding_dim,
                100,  # number of clusters
                self.faiss.METRIC_INNER_PRODUCT
            )
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        # Move to GPU if requested
        if self.use_gpu:
            try:
                res = self.faiss.StandardGpuResources()
                index = self.faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("Moved FAISS index to GPU")
            except Exception as e:
                logger.warning(f"Failed to move index to GPU: {e}. Using CPU.")

        return index

    def add(self, chunks: List[Chunk], embeddings: np.ndarray):
        """Add chunks with embeddings to the index.

        Args:
            chunks: List of chunks to add
            embeddings: Embeddings for the chunks (shape: [len(chunks), embedding_dim])
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) != "
                f"number of embeddings ({len(embeddings)})"
            )

        if len(chunks) == 0:
            logger.warning("No chunks to add to vector store")
            return

        # Ensure embeddings are float32 (FAISS requirement)
        embeddings = embeddings.astype(np.float32)

        # Normalize if requested
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-10)  # Avoid division by zero

        # For IVF index, need to train first
        if self.index_type == "IndexIVFFlat" and not self.index.is_trained:
            if self.debug:
                logger.debug("Training IVF index...")
            self.index.train(embeddings)

        # Add to index
        self.index.add(embeddings)

        # Store chunks
        self.chunks.extend(chunks)

        if self.debug:
            logger.debug(
                f"Added {len(chunks)} chunks to FAISS index "
                f"(total: {self.index.ntotal})"
            )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[SearchResult]:
        """Search for similar chunks.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of SearchResult objects, sorted by score (descending)
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty, cannot search")
            return []

        # Ensure query is 2D and float32
        query_embedding = query_embedding.astype(np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Normalize if needed
        if self.normalize:
            norm = np.linalg.norm(query_embedding)
            query_embedding = query_embedding / (norm + 1e-10)

        # Search
        top_k = min(top_k, self.index.ntotal)  # Can't retrieve more than we have
        scores, indices = self.index.search(query_embedding, top_k)

        # Convert to SearchResult objects
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < len(self.chunks):  # Valid index
                results.append(
                    SearchResult(
                        chunk=self.chunks[idx],
                        score=float(score),
                        rank=rank
                    )
                )

        if self.debug:
            logger.debug(
                f"Retrieved {len(results)} chunks "
                f"(scores: {[f'{r.score:.4f}' for r in results[:3]]}...)"
            )

        return results

    def save(self, path: Path):
        """Save vector store to disk.

        Args:
            path: Directory to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = path / "faiss.index"
        if self.use_gpu:
            # Move to CPU before saving
            cpu_index = self.faiss.index_gpu_to_cpu(self.index)
            self.faiss.write_index(cpu_index, str(index_path))
        else:
            self.faiss.write_index(self.index, str(index_path))

        # Save chunks
        chunks_path = path / "chunks.pkl"
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

        # Save metadata
        metadata = {
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "normalize": self.normalize,
            "num_vectors": self.index.ntotal,
            "num_chunks": len(self.chunks),
        }
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        if self.debug:
            logger.debug(f"Saved vector store to {path}")

    def load(self, path: Path):
        """Load vector store from disk.

        Args:
            path: Directory to load from
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Vector store not found at {path}")

        # Load metadata
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        # Validate
        if metadata["embedding_dim"] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"expected {self.embedding_dim}, got {metadata['embedding_dim']}"
            )

        # Load FAISS index
        index_path = path / "faiss.index"
        self.index = self.faiss.read_index(str(index_path))

        # Move to GPU if requested
        if self.use_gpu:
            try:
                res = self.faiss.StandardGpuResources()
                self.index = self.faiss.index_cpu_to_gpu(res, 0, self.index)
            except Exception as e:
                logger.warning(f"Failed to move loaded index to GPU: {e}")

        # Load chunks
        chunks_path = path / "chunks.pkl"
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        if self.debug:
            logger.debug(
                f"Loaded vector store from {path} "
                f"({metadata['num_vectors']} vectors, {metadata['num_chunks']} chunks)"
            )

    def get_num_vectors(self) -> int:
        """Get number of vectors in index."""
        return self.index.ntotal

    def clear(self):
        """Clear the vector store."""
        self.index.reset()
        self.chunks.clear()

        if self.debug:
            logger.debug("Cleared vector store")

    def __repr__(self):
        return (
            f"FAISSVectorStore(index_type={self.index_type}, "
            f"dim={self.embedding_dim}, "
            f"vectors={self.index.ntotal})"
        )


class VectorStoreFactory:
    """Factory for creating vector stores."""

    @staticmethod
    def create(
        store_type: str,
        embedding_dim: int,
        index_type: str = "IndexFlatIP",
        normalize: bool = True,
        use_gpu: bool = False,
        debug: bool = False
    ) -> VectorStore:
        """Create a vector store.

        Args:
            store_type: Type of store ('faiss')
            embedding_dim: Embedding dimension
            index_type: FAISS index type
            normalize: Normalize vectors
            use_gpu: Use GPU acceleration
            debug: Enable debug logging

        Returns:
            VectorStore instance
        """
        store_type = store_type.lower()

        if store_type == "faiss":
            return FAISSVectorStore(
                embedding_dim=embedding_dim,
                index_type=index_type,
                normalize=normalize,
                use_gpu=use_gpu,
                debug=debug
            )
        else:
            raise ValueError(
                f"Unknown vector store type: {store_type}. "
                f"Supported: 'faiss'"
            )
