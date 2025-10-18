"""
Main RAG pipeline orchestrator.

Coordinates chunking, embedding, indexing, retrieval, and reranking
for efficient context retrieval.
"""

import logging
import time
from typing import List, Optional
from pathlib import Path

from .rag_config import RAGConfig
from .embeddings import EmbeddingModel
from .chunking import ChunkerFactory, Chunk
from .vector_store import VectorStoreFactory, VectorStore
from .retriever import Reranker, RetrieverWithReranking
from .cache import RAGCache


logger = logging.getLogger(__name__)


class RAGPipeline:
    """Main RAG pipeline for context retrieval.

    This orchestrates the entire RAG process:
    1. Chunking: Split context into overlapping chunks
    2. Embedding: Generate embeddings for chunks
    3. Indexing: Store in vector database (FAISS)
    4. Retrieval: Find relevant chunks for queries
    5. Reranking: Reorder chunks by relevance (optional)
    """

    def __init__(self, config: RAGConfig):
        """Initialize RAG pipeline.

        Args:
            config: RAG configuration
        """
        self.config = config

        if not config.enabled:
            logger.warning("RAG pipeline initialized but disabled")
            return

        logger.info("Initializing RAG pipeline...")
        start_time = time.time()

        # Initialize components
        self.embedding_model = self._init_embedding_model()
        self.chunker = self._init_chunker()
        self.cache = self._init_cache()

        # Reranker (optional, loaded lazily)
        self.reranker = None
        if config.use_reranker:
            self.reranker = self._init_reranker()

        # Vector stores (one per context, loaded from cache or created)
        self.vector_stores = {}  # context_id -> VectorStore
        self.retrievers = {}  # context_id -> RetrieverWithReranking

        init_time = time.time() - start_time
        logger.info(f"RAG pipeline initialized in {init_time:.2f}s")

        if config.debug:
            self._print_config()

    def _init_embedding_model(self) -> EmbeddingModel:
        """Initialize embedding model."""
        return EmbeddingModel(
            model_name=self.config.embedding_model,
            batch_size=self.config.embedding_batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            debug=self.config.debug
        )

    def _init_chunker(self):
        """Initialize text chunker."""
        strategy = "sentence" if self.config.use_semantic_chunking else "fixed"

        return ChunkerFactory.create(
            strategy=strategy,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            add_metadata=self.config.add_metadata_to_chunks,
            debug=self.config.debug,
            print_chunks=self.config.print_chunks
        )

    def _init_cache(self) -> RAGCache:
        """Initialize cache manager."""
        return RAGCache(
            cache_dir=self.config.cache_dir,
            enabled=self.config.cache_enabled,
            debug=self.config.debug
        )

    def _init_reranker(self) -> Optional[Reranker]:
        """Initialize reranker (lazy loading)."""
        if not self.config.use_reranker:
            return None

        return Reranker(
            model_name=self.config.reranker_model,
            debug=self.config.debug,
            print_retrieval=self.config.print_retrieval
        )

    def index_context(self, context: str, context_id: Optional[str] = None) -> str:
        """Index a context for retrieval.

        This chunks the context, generates embeddings, and stores in
        a vector database. Results are cached for reuse.

        Args:
            context: Text to index
            context_id: Optional context identifier. If None, generated from hash.

        Returns:
            Context ID (hash) for later retrieval
        """
        if not self.config.enabled:
            raise RuntimeError("RAG pipeline is disabled")

        # Generate context ID if not provided
        if context_id is None:
            context_id = self.config.get_cache_key(context)

        if self.config.debug:
            logger.debug(f"Indexing context {context_id[:8]}... ({len(context)} chars)")

        # Check cache
        if self.cache.has_index(context_id):
            if self.config.debug:
                logger.debug(f"Loading index from cache: {context_id[:8]}...")

            # Load from cache
            vector_store = self._load_vector_store_from_cache(context_id)
            self.vector_stores[context_id] = vector_store

            # Create retriever
            retriever = self._create_retriever(vector_store)
            self.retrievers[context_id] = retriever

            return context_id

        # Not in cache, need to index
        if self.config.debug:
            logger.debug(f"Indexing new context: {context_id[:8]}...")

        start_time = time.time()

        # 1. Chunk the context
        chunk_start = time.time()
        chunks = self.chunker.chunk(context, context_id)
        chunk_time = time.time() - chunk_start

        if not chunks:
            logger.warning("No chunks generated from context")
            return context_id

        if self.config.debug:
            logger.debug(
                f"Generated {len(chunks)} chunks in {chunk_time:.2f}s "
                f"(avg {len(context) / len(chunks):.0f} chars/chunk)"
            )

        # 2. Generate embeddings
        embed_start = time.time()
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.embed_batch(
            chunk_texts,
            show_progress=self.config.debug
        )
        embed_time = time.time() - embed_start

        if self.config.debug:
            logger.debug(
                f"Generated {len(embeddings)} embeddings in {embed_time:.2f}s "
                f"({len(embeddings) / embed_time:.1f} embeddings/sec)"
            )

        # 3. Create vector store and add embeddings
        vector_store = VectorStoreFactory.create(
            store_type=self.config.vector_store_type,
            embedding_dim=self.embedding_model.get_embedding_dim(),
            index_type=self.config.faiss_index_type,
            normalize=self.config.normalize_embeddings,
            use_gpu=self.config.use_gpu,
            debug=self.config.debug
        )

        vector_store.add(chunks, embeddings)
        self.vector_stores[context_id] = vector_store

        # 4. Save to cache
        if self.config.cache_enabled:
            cache_path = self.cache.get_index_path(context_id)
            vector_store.save(cache_path)

            # Save metadata
            metadata = {
                "context_id": context_id,
                "context_length": len(context),
                "num_chunks": len(chunks),
                "num_vectors": vector_store.get_num_vectors(),
                "embedding_model": self.config.embedding_model,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
            }
            self.cache.save_metadata(context_id, metadata)

        # 5. Create retriever
        retriever = self._create_retriever(vector_store)
        self.retrievers[context_id] = retriever

        index_time = time.time() - start_time

        if self.config.log_retrieval_stats:
            logger.info(
                f"Indexed context {context_id[:8]} in {index_time:.2f}s "
                f"({len(chunks)} chunks, {len(embeddings)} vectors)"
            )

        return context_id

    def retrieve(
        self,
        question: str,
        context_id: str,
        top_k: Optional[int] = None
    ) -> str:
        """Retrieve relevant chunks for a question.

        Args:
            question: Query text
            context_id: Context identifier (from index_context)
            top_k: Number of chunks to retrieve (uses config default if None)

        Returns:
            Retrieved context as concatenated chunks
        """
        if not self.config.enabled:
            raise RuntimeError("RAG pipeline is disabled")

        if context_id not in self.retrievers:
            raise ValueError(
                f"Context {context_id[:8]} not indexed. "
                f"Call index_context() first."
            )

        if self.config.debug:
            logger.debug(f"Retrieving chunks for question: {question[:100]}...")

        start_time = time.time()

        # Get retriever
        retriever = self.retrievers[context_id]

        # Retrieve and rerank
        results = retriever.retrieve_and_rerank(question)

        # Expand with neighboring chunks if configured
        if self.config.context_window_expansion > 0:
            results = self._expand_context_window(results, context_id)

        # Assemble retrieved context
        retrieved_context = self._assemble_context(results)

        retrieve_time = time.time() - start_time

        if self.config.log_retrieval_stats:
            logger.info(
                f"Retrieved {len(results)} chunks in {retrieve_time:.3f}s "
                f"({len(retrieved_context)} chars)"
            )

        return retrieved_context

    def _load_vector_store_from_cache(self, context_id: str) -> VectorStore:
        """Load vector store from cache."""
        cache_path = self.cache.get_index_path(context_id)

        vector_store = VectorStoreFactory.create(
            store_type=self.config.vector_store_type,
            embedding_dim=self.embedding_model.get_embedding_dim(),
            index_type=self.config.faiss_index_type,
            normalize=self.config.normalize_embeddings,
            use_gpu=self.config.use_gpu,
            debug=self.config.debug
        )

        vector_store.load(cache_path)

        return vector_store

    def _create_retriever(self, vector_store: VectorStore) -> RetrieverWithReranking:
        """Create retriever for a vector store."""
        return RetrieverWithReranking(
            vector_store=vector_store,
            embedding_model=self.embedding_model,
            reranker=self.reranker,
            top_k=self.config.top_k,
            top_k_rerank=self.config.top_k_rerank,
            debug=self.config.debug,
            print_retrieval=self.config.print_retrieval
        )

    def _expand_context_window(self, results, context_id: str):
        """Expand retrieved chunks with neighboring chunks."""
        # TODO: Implement context window expansion
        # This would retrieve neighboring chunks for more context
        return results

    def _assemble_context(self, results) -> str:
        """Assemble retrieved chunks into a single context string."""
        if not results:
            return ""

        # Sort by chunk position (not by score) for coherent reading
        sorted_results = sorted(results, key=lambda r: r.chunk.chunk_id)

        # Concatenate chunks
        chunks_text = []
        for result in sorted_results:
            chunks_text.append(result.chunk.text)

        # Join with newlines
        context = "\n\n".join(chunks_text)

        return context

    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        stats = {
            "config": self.config.to_dict(),
            "cache": self.cache.get_stats() if self.config.cache_enabled else {},
            "num_indexed_contexts": len(self.vector_stores),
            "indexed_contexts": list(self.vector_stores.keys()),
        }

        # Add retriever stats if available
        if self.retrievers:
            retriever_stats = {}
            for context_id, retriever in self.retrievers.items():
                retriever_stats[context_id[:8]] = retriever.get_stats()
            stats["retriever_stats"] = retriever_stats

        return stats

    def print_stats(self):
        """Print pipeline statistics."""
        stats = self.get_stats()

        print(f"\n{'='*80}")
        print("RAG PIPELINE STATISTICS")
        print(f"{'='*80}")
        print(f"Enabled: {self.config.enabled}")
        print(f"Embedding model: {self.config.embedding_model}")
        print(f"Chunk size: {self.config.chunk_size} tokens")
        print(f"Chunk overlap: {self.config.chunk_overlap} tokens")
        print(f"Top-k retrieval: {self.config.top_k}")
        print(f"Top-k rerank: {self.config.top_k_rerank}")
        print(f"Reranker: {self.config.reranker_model if self.config.use_reranker else 'None'}")
        print(f"\nIndexed contexts: {stats['num_indexed_contexts']}")

        if self.config.cache_enabled:
            cache_stats = stats['cache']
            print(f"\nCache:")
            print(f"  Indices: {cache_stats['num_indices']}")
            print(f"  Size: {cache_stats['cache_size_mb']:.2f} MB")
            print(f"  Hit rate: {cache_stats['hit_rate']:.2%}")

        print(f"{'='*80}\n")

    def _print_config(self):
        """Print configuration (debug)."""
        print(f"\n{'='*80}")
        print("RAG PIPELINE CONFIGURATION")
        print(f"{'='*80}")
        print(f"Enabled: {self.config.enabled}")
        print(f"Embedding: {self.config.embedding_model}")
        print(f"Chunking: {'semantic' if self.config.use_semantic_chunking else 'fixed'}")
        print(f"  Size: {self.config.chunk_size} tokens")
        print(f"  Overlap: {self.config.chunk_overlap} tokens")
        print(f"Vector store: {self.config.vector_store_type}")
        print(f"  Index type: {self.config.faiss_index_type}")
        print(f"  GPU: {self.config.use_gpu}")
        print(f"Retrieval:")
        print(f"  Top-k: {self.config.top_k}")
        print(f"  Rerank: {self.config.use_reranker}")
        if self.config.use_reranker:
            print(f"  Reranker: {self.config.reranker_model}")
            print(f"  Top-k rerank: {self.config.top_k_rerank}")
        print(f"Cache:")
        print(f"  Enabled: {self.config.cache_enabled}")
        print(f"  Directory: {self.config.cache_dir}")
        print(f"Debug:")
        print(f"  Debug logging: {self.config.debug}")
        print(f"  Print chunks: {self.config.print_chunks}")
        print(f"  Print retrieval: {self.config.print_retrieval}")
        print(f"{'='*80}\n")

    def clear_cache(self):
        """Clear the RAG cache."""
        self.cache.clear()

    def __repr__(self):
        return (
            f"RAGPipeline(enabled={self.config.enabled}, "
            f"embedding={self.config.embedding_model}, "
            f"indexed_contexts={len(self.vector_stores)})"
        )
