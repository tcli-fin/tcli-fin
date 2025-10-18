"""
Cache management for RAG pipeline.

Handles caching of vector indices to avoid recomputing embeddings
for the same contexts.
"""

import logging
import json
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime


logger = logging.getLogger(__name__)


class RAGCache:
    """Cache manager for RAG indices.

    Stores vector indices and metadata on disk, keyed by context hash.
    This avoids recomputing embeddings for the same contexts.
    """

    def __init__(
        self,
        cache_dir: str = ".rag_cache",
        enabled: bool = True,
        debug: bool = False
    ):
        """Initialize cache manager.

        Args:
            cache_dir: Directory for cache storage
            enabled: Whether caching is enabled
            debug: Enable debug logging
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.debug = debug

        # Create cache directory
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            if self.debug:
                logger.debug(f"RAG cache directory: {self.cache_dir.absolute()}")

        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0

    def has_index(self, context_id: str) -> bool:
        """Check if an index exists for a context.

        Args:
            context_id: Context identifier (hash)

        Returns:
            True if index exists in cache
        """
        if not self.enabled:
            return False

        index_path = self._get_index_path(context_id)
        exists = index_path.exists()

        if self.debug:
            logger.debug(
                f"Cache {'HIT' if exists else 'MISS'} for context {context_id[:8]}..."
            )

        if exists:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        return exists

    def get_index_path(self, context_id: str) -> Path:
        """Get path to cached index.

        Args:
            context_id: Context identifier

        Returns:
            Path to index directory
        """
        return self._get_index_path(context_id)

    def save_metadata(self, context_id: str, metadata: dict):
        """Save metadata for a cached index.

        Args:
            context_id: Context identifier
            metadata: Metadata dictionary to save
        """
        if not self.enabled:
            return

        index_path = self._get_index_path(context_id)

        # Ensure directory exists
        index_path.mkdir(parents=True, exist_ok=True)

        metadata_path = index_path / "cache_metadata.json"

        # Add timestamp
        metadata["cached_at"] = datetime.now().isoformat()

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        if self.debug:
            logger.debug(f"Saved cache metadata for {context_id[:8]}...")

    def load_metadata(self, context_id: str) -> Optional[dict]:
        """Load metadata for a cached index.

        Args:
            context_id: Context identifier

        Returns:
            Metadata dictionary, or None if not found
        """
        if not self.enabled:
            return None

        index_path = self._get_index_path(context_id)
        metadata_path = index_path / "cache_metadata.json"

        if not metadata_path.exists():
            return None

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        if self.debug:
            logger.debug(f"Loaded cache metadata for {context_id[:8]}...")

        return metadata

    def invalidate(self, context_id: str):
        """Invalidate (delete) a cached index.

        Args:
            context_id: Context identifier
        """
        if not self.enabled:
            return

        index_path = self._get_index_path(context_id)

        if index_path.exists():
            shutil.rmtree(index_path)

            if self.debug:
                logger.debug(f"Invalidated cache for {context_id[:8]}...")

    def clear(self):
        """Clear entire cache."""
        if not self.enabled:
            return

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Cleared RAG cache")

    def get_size(self) -> int:
        """Get cache size in bytes.

        Returns:
            Total size of cache directory in bytes
        """
        if not self.enabled or not self.cache_dir.exists():
            return 0

        total_size = 0
        for path in self.cache_dir.rglob("*"):
            if path.is_file():
                total_size += path.stat().st_size

        return total_size

    def get_num_indices(self) -> int:
        """Get number of cached indices.

        Returns:
            Number of cached indices
        """
        if not self.enabled or not self.cache_dir.exists():
            return 0

        # Count directories (each is an index)
        return len([d for d in self.cache_dir.iterdir() if d.is_dir()])

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

        return {
            "enabled": self.enabled,
            "cache_dir": str(self.cache_dir.absolute()),
            "num_indices": self.get_num_indices(),
            "cache_size_bytes": self.get_size(),
            "cache_size_mb": self.get_size() / (1024 * 1024),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
        }

    def print_stats(self):
        """Print cache statistics."""
        stats = self.get_stats()

        print(f"\n{'='*60}")
        print("RAG CACHE STATISTICS")
        print(f"{'='*60}")
        print(f"Enabled: {stats['enabled']}")
        print(f"Directory: {stats['cache_dir']}")
        print(f"Cached indices: {stats['num_indices']}")
        print(f"Cache size: {stats['cache_size_mb']:.2f} MB")
        print(f"Cache hits: {stats['cache_hits']}")
        print(f"Cache misses: {stats['cache_misses']}")
        print(f"Hit rate: {stats['hit_rate']:.2%}")
        print(f"{'='*60}\n")

    def list_indices(self) -> list:
        """List all cached indices.

        Returns:
            List of tuples (context_id, metadata)
        """
        if not self.enabled or not self.cache_dir.exists():
            return []

        indices = []
        for index_dir in self.cache_dir.iterdir():
            if index_dir.is_dir():
                context_id = index_dir.name
                metadata = self.load_metadata(context_id)
                indices.append((context_id, metadata))

        return indices

    def _get_index_path(self, context_id: str) -> Path:
        """Get path for storing index.

        Args:
            context_id: Context identifier

        Returns:
            Path to index directory
        """
        return self.cache_dir / context_id
