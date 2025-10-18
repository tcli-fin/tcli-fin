"""
Text chunking strategies for RAG pipeline.

Provides different strategies for splitting long texts into chunks
that can be embedded and retrieved.
"""

import logging
import re
from typing import List, Optional, Tuple
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text chunk with metadata."""

    text: str
    """The chunk text."""

    chunk_id: int
    """Index of this chunk in the document."""

    start_char: int
    """Starting character position in original text."""

    end_char: int
    """Ending character position in original text."""

    tokens: Optional[int] = None
    """Approximate token count (if computed)."""

    metadata: dict = None
    """Additional metadata about the chunk."""

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def __repr__(self):
        return (
            f"Chunk(id={self.chunk_id}, chars={self.start_char}-{self.end_char}, "
            f"tokens={self.tokens}, len={len(self.text)})"
        )


class TextChunker:
    """Base class for text chunking strategies."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        add_metadata: bool = True,
        debug: bool = False,
        print_chunks: bool = False
    ):
        """Initialize chunker.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            add_metadata: Add metadata to chunks
            debug: Enable debug logging
            print_chunks: Print chunks when created (for debugging)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.add_metadata = add_metadata
        self.debug = debug
        self.print_chunks = print_chunks

    def chunk(self, text: str, context_id: Optional[str] = None) -> List[Chunk]:
        """Chunk text into overlapping segments.

        Args:
            text: Text to chunk
            context_id: Optional context identifier for metadata

        Returns:
            List of Chunk objects
        """
        raise NotImplementedError("Subclasses must implement chunk()")

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (very rough approximation).

        Uses a simple heuristic: ~4 characters per token on average.
        This is rough but avoids loading a tokenizer.
        """
        return len(text) // 4

    def _print_chunk_debug(self, chunk: Chunk, total_chunks: int):
        """Print chunk for debugging."""
        if self.print_chunks:
            print(f"\n{'='*80}")
            print(f"Chunk {chunk.chunk_id + 1}/{total_chunks}")
            print(f"  Position: chars {chunk.start_char}-{chunk.end_char}")
            print(f"  Length: {len(chunk.text)} chars, ~{chunk.tokens} tokens")
            print(f"  Preview: {chunk.text[:200]}...")
            print(f"{'='*80}\n")


class FixedSizeChunker(TextChunker):
    """Chunk text into fixed-size chunks with overlap.

    This is the simplest and fastest chunking strategy.
    Chunks are created by splitting on character boundaries with overlap.
    """

    def chunk(self, text: str, context_id: Optional[str] = None) -> List[Chunk]:
        """Chunk text into fixed-size overlapping chunks."""

        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []

        # Convert token sizes to approximate character sizes
        chunk_chars = self.chunk_size * 4  # ~4 chars per token
        overlap_chars = self.chunk_overlap * 4

        chunks = []
        start = 0
        chunk_id = 0

        if self.debug:
            logger.debug(
                f"Chunking text of {len(text)} chars into chunks of "
                f"~{chunk_chars} chars with {overlap_chars} char overlap"
            )

        while start < len(text):
            end = min(start + chunk_chars, len(text))

            # Extract chunk
            chunk_text = text[start:end].strip()

            if chunk_text:  # Skip empty chunks
                chunk = Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    start_char=start,
                    end_char=end,
                    tokens=self._estimate_tokens(chunk_text),
                    metadata={
                        "context_id": context_id,
                        "chunker": "fixed_size",
                    } if self.add_metadata else {}
                )

                chunks.append(chunk)
                self._print_chunk_debug(chunk, -1)  # Don't know total yet
                chunk_id += 1

            # Move to next chunk with overlap
            # CRITICAL FIX: Ensure we're actually advancing forward
            new_start = end - overlap_chars

            # If we're not advancing or going backwards, we're done
            if new_start <= start:
                break

            start = new_start

        if self.debug:
            logger.debug(f"Created {len(chunks)} chunks from text")

        # Update total chunks in debug output
        if self.print_chunks:
            print(f"\n{'='*80}")
            print(f"TOTAL: {len(chunks)} chunks created")
            print(f"{'='*80}\n")

        return chunks


class SentenceChunker(TextChunker):
    """Chunk text by sentences with size limits.

    This creates more semantically coherent chunks by respecting
    sentence boundaries. Chunks are built up to the target size
    without breaking sentences.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Simple sentence splitter pattern
        # Matches period, exclamation, question mark followed by space/newline
        self.sentence_pattern = re.compile(r'([.!?]+[\s\n]+)')

    def chunk(self, text: str, context_id: Optional[str] = None) -> List[Chunk]:
        """Chunk text by sentences."""

        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []

        # Split into sentences
        sentences = self._split_sentences(text)

        if self.debug:
            logger.debug(f"Split text into {len(sentences)} sentences")

        # Build chunks from sentences
        chunks = []
        current_chunk_sentences = []
        current_chunk_tokens = 0
        chunk_id = 0

        for sentence, start_char, end_char in sentences:
            sentence_tokens = self._estimate_tokens(sentence)

            # Check if adding this sentence would exceed chunk size
            if (current_chunk_tokens + sentence_tokens > self.chunk_size
                and current_chunk_sentences):  # Don't create empty chunks

                # Create chunk from accumulated sentences
                chunk = self._create_chunk_from_sentences(
                    current_chunk_sentences,
                    chunk_id,
                    context_id
                )
                chunks.append(chunk)
                self._print_chunk_debug(chunk, -1)
                chunk_id += 1

                # Start new chunk with overlap
                # Keep last few sentences for overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk_sentences,
                    self.chunk_overlap
                )
                current_chunk_sentences = overlap_sentences
                current_chunk_tokens = sum(
                    self._estimate_tokens(s[0]) for s in overlap_sentences
                )

            # Add sentence to current chunk
            current_chunk_sentences.append((sentence, start_char, end_char))
            current_chunk_tokens += sentence_tokens

        # Don't forget the last chunk
        if current_chunk_sentences:
            chunk = self._create_chunk_from_sentences(
                current_chunk_sentences,
                chunk_id,
                context_id
            )
            chunks.append(chunk)
            self._print_chunk_debug(chunk, len(chunks))

        if self.debug:
            logger.debug(f"Created {len(chunks)} sentence-based chunks")

        if self.print_chunks:
            print(f"\n{'='*80}")
            print(f"TOTAL: {len(chunks)} sentence-based chunks created")
            print(f"{'='*80}\n")

        return chunks

    def _split_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text into sentences with character positions.

        Returns:
            List of (sentence_text, start_char, end_char)
        """
        sentences = []
        last_end = 0

        for match in self.sentence_pattern.finditer(text):
            end = match.end()
            sentence = text[last_end:end].strip()

            if sentence:
                sentences.append((sentence, last_end, end))

            last_end = end

        # Don't forget text after last sentence marker
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining:
                sentences.append((remaining, last_end, len(text)))

        return sentences

    def _create_chunk_from_sentences(
        self,
        sentences: List[Tuple[str, int, int]],
        chunk_id: int,
        context_id: Optional[str]
    ) -> Chunk:
        """Create a chunk from a list of sentences."""
        chunk_text = " ".join(s[0] for s in sentences)
        start_char = sentences[0][1]
        end_char = sentences[-1][2]

        return Chunk(
            text=chunk_text,
            chunk_id=chunk_id,
            start_char=start_char,
            end_char=end_char,
            tokens=self._estimate_tokens(chunk_text),
            metadata={
                "context_id": context_id,
                "chunker": "sentence",
                "num_sentences": len(sentences),
            } if self.add_metadata else {}
        )

    def _get_overlap_sentences(
        self,
        sentences: List[Tuple[str, int, int]],
        target_overlap_tokens: int
    ) -> List[Tuple[str, int, int]]:
        """Get sentences for overlap from end of chunk."""
        overlap_sentences = []
        overlap_tokens = 0

        # Take sentences from end until we reach target overlap
        for sentence in reversed(sentences):
            sentence_tokens = self._estimate_tokens(sentence[0])
            if overlap_tokens + sentence_tokens > target_overlap_tokens:
                break
            overlap_sentences.insert(0, sentence)
            overlap_tokens += sentence_tokens

        return overlap_sentences


class ChunkerFactory:
    """Factory for creating chunkers."""

    @staticmethod
    def create(
        strategy: str = "fixed",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        add_metadata: bool = True,
        debug: bool = False,
        print_chunks: bool = False
    ) -> TextChunker:
        """Create a chunker.

        Args:
            strategy: Chunking strategy ('fixed' or 'sentence')
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap in tokens
            add_metadata: Add metadata to chunks
            debug: Enable debug logging
            print_chunks: Print chunks for debugging

        Returns:
            TextChunker instance
        """
        strategy = strategy.lower()

        if strategy == "fixed":
            return FixedSizeChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_metadata=add_metadata,
                debug=debug,
                print_chunks=print_chunks
            )
        elif strategy in ["sentence", "semantic"]:
            return SentenceChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_metadata=add_metadata,
                debug=debug,
                print_chunks=print_chunks
            )
        else:
            raise ValueError(
                f"Unknown chunking strategy: {strategy}. "
                f"Use 'fixed' or 'sentence'."
            )
