"""
Retrieval and reranking for RAG pipeline.

Handles searching the vector store and optionally reranking results
using a cross-encoder model.
"""

import logging
import time
import numpy as np
from typing import List, Optional

from .vector_store import VectorStore, SearchResult
from .embeddings import EmbeddingModel


logger = logging.getLogger(__name__)


class Retriever:
    """Retriever for finding relevant chunks."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        top_k: int = 20,
        debug: bool = False,
        print_retrieval: bool = False
    ):
        """Initialize retriever.

        Args:
            vector_store: Vector store to search
            embedding_model: Model for embedding queries
            top_k: Number of chunks to retrieve
            debug: Enable debug logging
            print_retrieval: Print retrieval results (for debugging)
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.debug = debug
        self.print_retrieval = print_retrieval

    def retrieve(self, query: str) -> List[SearchResult]:
        """Retrieve relevant chunks for a query.

        Args:
            query: Query text

        Returns:
            List of SearchResult objects
        """
        if self.debug:
            logger.debug(f"Retrieving chunks for query: {query[:100]}...")

        start_time = time.time()

        # Embed query
        query_embedding = self.embedding_model.embed(query)

        # Search vector store
        results = self.vector_store.search(query_embedding, self.top_k)

        elapsed = time.time() - start_time

        if self.debug:
            logger.debug(
                f"Retrieved {len(results)} chunks in {elapsed:.3f}s "
                f"(avg score: {np.mean([r.score for r in results]):.4f})"
            )

        # Print retrieval details if requested
        if self.print_retrieval:
            self._print_retrieval_results(query, results)

        return results

    def _print_retrieval_results(self, query: str, results: List[SearchResult]):
        """Print detailed retrieval results for debugging."""
        print(f"\n{'='*80}")
        print(f"RETRIEVAL RESULTS")
        print(f"{'='*80}")
        print(f"Query: {query}")
        print(f"Retrieved: {len(results)} chunks\n")

        for i, result in enumerate(results[:10], 1):  # Show top 10
            print(f"\n--- Rank {i} (Score: {result.score:.4f}) ---")
            print(f"Chunk ID: {result.chunk.chunk_id}")
            print(f"Position: chars {result.chunk.start_char}-{result.chunk.end_char}")
            print(f"Tokens: ~{result.chunk.tokens}")
            print(f"Preview: {result.chunk.text[:200]}...")

        print(f"\n{'='*80}\n")


class Reranker:
    """Cross-encoder reranker for improving retrieval quality."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        debug: bool = False,
        print_retrieval: bool = False
    ):
        """Initialize reranker.

        Args:
            model_name: Cross-encoder model name
            batch_size: Batch size for reranking
            debug: Enable debug logging
            print_retrieval: Print reranking results
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.debug = debug
        self.print_retrieval = print_retrieval

        logger.info(f"Loading reranker model: {model_name}")

        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        start_time = time.time()
        self.model = CrossEncoder(model_name)
        load_time = time.time() - start_time

        logger.info(f"Loaded reranker in {load_time:.2f}s")

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """Rerank search results using cross-encoder.

        Args:
            query: Query text
            results: Initial search results
            top_k: Number of results to return after reranking

        Returns:
            Reranked results (top_k only)
        """
        if not results:
            return []

        if self.debug:
            logger.debug(f"Reranking {len(results)} results to top-{top_k}")

        start_time = time.time()

        # Prepare query-chunk pairs
        pairs = [(query, result.chunk.text) for result in results]

        # Score with cross-encoder
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        # Create new results with reranker scores
        reranked = []
        for result, score in zip(results, scores):
            # Create new SearchResult with updated score
            reranked.append(
                SearchResult(
                    chunk=result.chunk,
                    score=float(score),
                    rank=-1  # Will be updated below
                )
            )

        # Sort by new scores (descending)
        reranked.sort(key=lambda x: x.score, reverse=True)

        # Update ranks and take top_k
        final_results = []
        for rank, result in enumerate(reranked[:top_k]):
            final_results.append(
                SearchResult(
                    chunk=result.chunk,
                    score=result.score,
                    rank=rank
                )
            )

        elapsed = time.time() - start_time

        if self.debug:
            logger.debug(
                f"Reranked {len(results)} -> {len(final_results)} in {elapsed:.3f}s "
                f"(avg score: {np.mean([r.score for r in final_results]):.4f})"
            )

        # Print reranking details if requested
        if self.print_retrieval:
            self._print_reranking_results(query, results, final_results)

        return final_results

    def _print_reranking_results(
        self,
        query: str,
        before: List[SearchResult],
        after: List[SearchResult]
    ):
        """Print reranking comparison for debugging."""
        print(f"\n{'='*80}")
        print(f"RERANKING RESULTS")
        print(f"{'='*80}")
        print(f"Query: {query}")
        print(f"Before reranking: {len(before)} chunks")
        print(f"After reranking: {len(after)} chunks\n")

        print("Score changes:")
        # Show score changes for chunks that made it to final results
        for result_after in after[:5]:  # Top 5
            # Find this chunk in before results
            chunk_id = result_after.chunk.chunk_id
            result_before = next(
                (r for r in before if r.chunk.chunk_id == chunk_id),
                None
            )

            if result_before:
                print(
                    f"  Chunk {chunk_id}: "
                    f"rank {result_before.rank} -> {result_after.rank}, "
                    f"score {result_before.score:.4f} -> {result_after.score:.4f}"
                )

        print(f"\n{'='*80}\n")


class RetrieverWithReranking:
    """Retriever that combines vector search with reranking."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        reranker: Optional[Reranker] = None,
        top_k: int = 20,
        top_k_rerank: int = 5,
        debug: bool = False,
        print_retrieval: bool = False
    ):
        """Initialize retriever with reranking.

        Args:
            vector_store: Vector store to search
            embedding_model: Model for embedding queries
            reranker: Reranker model (optional)
            top_k: Number of chunks to retrieve initially
            top_k_rerank: Number of chunks after reranking
            debug: Enable debug logging
            print_retrieval: Print retrieval results
        """
        self.retriever = Retriever(
            vector_store=vector_store,
            embedding_model=embedding_model,
            top_k=top_k,
            debug=debug,
            print_retrieval=print_retrieval
        )
        self.reranker = reranker
        self.top_k_rerank = top_k_rerank
        self.debug = debug
        self.print_retrieval = print_retrieval

        # Statistics
        self.total_queries = 0
        self.total_retrieval_time = 0.0
        self.total_rerank_time = 0.0

    def retrieve_and_rerank(self, query: str) -> List[SearchResult]:
        """Retrieve and optionally rerank chunks.

        Args:
            query: Query text

        Returns:
            List of SearchResult objects (reranked if reranker available)
        """
        start_time = time.time()

        # Initial retrieval
        results = self.retriever.retrieve(query)
        retrieval_time = time.time() - start_time
        self.total_retrieval_time += retrieval_time

        # Rerank if reranker available
        if self.reranker:
            rerank_start = time.time()
            results = self.reranker.rerank(query, results, self.top_k_rerank)
            rerank_time = time.time() - rerank_start
            self.total_rerank_time += rerank_time
        else:
            # No reranking, just take top_k_rerank
            results = results[:self.top_k_rerank]

        self.total_queries += 1

        total_time = time.time() - start_time

        if self.debug:
            logger.debug(
                f"Retrieve + rerank completed in {total_time:.3f}s "
                f"(retrieval: {retrieval_time:.3f}s, "
                f"rerank: {self.total_rerank_time / self.total_queries:.3f}s avg)"
            )

        return results

    def get_stats(self) -> dict:
        """Get retrieval statistics."""
        return {
            "total_queries": self.total_queries,
            "avg_retrieval_time": (
                self.total_retrieval_time / self.total_queries
                if self.total_queries > 0 else 0
            ),
            "avg_rerank_time": (
                self.total_rerank_time / self.total_queries
                if self.total_queries > 0 else 0
            ),
            "total_time": self.total_retrieval_time + self.total_rerank_time,
        }

    def reset_stats(self):
        """Reset statistics."""
        self.total_queries = 0
        self.total_retrieval_time = 0.0
        self.total_rerank_time = 0.0
