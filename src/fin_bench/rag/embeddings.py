"""
Embedding model wrapper for RAG pipeline.

Handles loading and using sentence transformer models for embedding generation.
"""

import logging
import numpy as np
from typing import List, Union, Optional
import time


logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wrapper for sentence transformer embedding models.

    Handles model loading, embedding generation, and batching.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        device: Optional[str] = None,
        debug: bool = False
    ):
        """Initialize embedding model.

        Args:
            model_name: HuggingFace model name
            batch_size: Batch size for embedding generation
            normalize_embeddings: Whether to normalize embeddings to unit length
            device: Device to use (cuda, cpu, mps). Auto-detected if None.
            debug: Enable debug logging
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.debug = debug

        logger.info(f"Loading embedding model: {model_name}")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        # Load model
        start_time = time.time()
        self.model = SentenceTransformer(model_name, device=device)
        load_time = time.time() - start_time

        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(
            f"Loaded {model_name} in {load_time:.2f}s "
            f"(dim={self.embedding_dim}, device={self.model.device})"
        )

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed
            show_progress: Show progress bar

        Returns:
            Array of embeddings (shape: [len(texts), embedding_dim])
        """
        if not texts:
            return np.array([])

        if self.debug:
            logger.debug(f"Embedding {len(texts)} texts (batch_size={self.batch_size})")

        start_time = time.time()

        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress and self.debug,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True
        )

        elapsed = time.time() - start_time

        if self.debug:
            logger.debug(
                f"Generated {len(texts)} embeddings in {elapsed:.2f}s "
                f"({len(texts)/elapsed:.1f} texts/sec)"
            )

        return embeddings

    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim

    def __repr__(self):
        return (
            f"EmbeddingModel(model={self.model_name}, "
            f"dim={self.embedding_dim}, "
            f"device={self.model.device})"
        )


class CachedEmbeddingModel(EmbeddingModel):
    """Embedding model with in-memory caching.

    Caches embeddings for texts to avoid recomputation.
    Useful when the same texts are embedded multiple times.
    """

    def __init__(self, *args, cache_size: int = 10000, **kwargs):
        """Initialize with cache.

        Args:
            cache_size: Maximum number of embeddings to cache
            *args, **kwargs: Passed to EmbeddingModel
        """
        super().__init__(*args, **kwargs)
        self.cache_size = cache_size
        self.cache = {}  # text -> embedding
        self.cache_hits = 0
        self.cache_misses = 0

    def embed(self, text: str) -> np.ndarray:
        """Embed with caching."""
        if text in self.cache:
            self.cache_hits += 1
            return self.cache[text]

        self.cache_misses += 1
        embedding = super().embed(text)

        # Add to cache (with simple eviction if full)
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            self.cache.pop(next(iter(self.cache)))

        self.cache[text] = embedding
        return embedding

    def embed_batch(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Embed batch with caching."""
        # Check which texts are cached
        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if text in self.cache:
                cached_embeddings[i] = self.cache[text]
                self.cache_hits += 1
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                self.cache_misses += 1

        # Embed uncached texts
        if uncached_texts:
            new_embeddings = super().embed_batch(uncached_texts, show_progress)

            # Add to cache
            for text, embedding in zip(uncached_texts, new_embeddings):
                if len(self.cache) >= self.cache_size:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[text] = embedding

            # Combine cached and new embeddings
            all_embeddings = np.zeros((len(texts), self.embedding_dim))
            for i, emb in cached_embeddings.items():
                all_embeddings[i] = emb
            for i, emb in zip(uncached_indices, new_embeddings):
                all_embeddings[i] = emb

            return all_embeddings
        else:
            # All cached
            return np.array([cached_embeddings[i] for i in range(len(texts))])

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
        }

    def clear_cache(self):
        """Clear embedding cache."""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
