# RAG Pipeline for FinBench

This document describes the Retrieval-Augmented Generation (RAG) pipeline implementation in FinBench.

## Overview

The RAG pipeline allows models without CLI tools to efficiently work with long financial documents by:
1. **Chunking** documents into smaller, semantically meaningful pieces
2. **Embedding** chunks using state-of-the-art embedding models
3. **Storing** embeddings in a fast vector database (FAISS)
4. **Retrieving** the most relevant chunks for each question
5. **Reranking** retrieved chunks for better quality (optional)
6. **Caching** indices to avoid recomputation

## Installation

Install RAG dependencies:

```bash
pip install -r requirements-rag.txt
```

For GPU acceleration (optional):
```bash
pip install faiss-gpu
```

## Quick Start

### 1. Test the RAG Pipeline

Run the test script to validate everything works:

```bash
python test_rag_pipeline.py
```

This will test:
- Basic RAG functionality
- Different chunking strategies
- Embedding generation
- Cache management

### 2. Run a RAG Experiment

Use the pre-configured RAG experiments:

```bash
# Quick test with debug output (1 sample)
finbench run docfinqa_rag_debug

# Small test (10 samples)
finbench run docfinqa_rag_test

# Fast RAG configuration (20 samples)
finbench run docfinqa_rag_fast

# Full evaluation with default RAG
finbench run docfinqa_rag_full
```

## Configuration

RAG configurations are defined in `config.yaml` under the `rag_configs` section.

### Available RAG Configurations

#### 1. `default_rag` - Best Quality
```yaml
default_rag:
  enabled: true
  embedding_model: "BAAI/bge-large-en-v1.5"  # Best quality embeddings
  chunk_size: 512
  chunk_overlap: 50
  top_k: 20  # Retrieve 20 chunks initially
  top_k_rerank: 5  # Rerank to top 5
  use_reranker: true  # Enable cross-encoder reranking
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

**Use when**: You want the best retrieval quality and accuracy is more important than speed.

#### 2. `fast_rag` - Fast Evaluation
```yaml
fast_rag:
  enabled: true
  embedding_model: "BAAI/bge-small-en-v1.5"  # Smaller, faster model
  chunk_size: 256
  chunk_overlap: 25
  top_k: 10
  top_k_rerank: 3
  use_reranker: false  # Skip reranking for speed
```

**Use when**: You want faster evaluation and can tolerate slightly lower quality.

#### 3. `semantic_rag` - Semantic Chunking
```yaml
semantic_rag:
  enabled: true
  embedding_model: "BAAI/bge-large-en-v1.5"
  use_semantic_chunking: true  # Use sentence boundaries
  # ... other settings
```

**Use when**: You want chunks to respect sentence boundaries for better semantic coherence.

#### 4. `debug_rag` - Debugging
```yaml
debug_rag:
  enabled: true
  debug: true
  print_chunks: true  # Print all chunks when indexing
  print_retrieval: true  # Print retrieval results
  # ... other settings
```

**Use when**: You want to see detailed information about chunking and retrieval for debugging.

## Creating Custom RAG Experiments

Add a new experiment in `config.yaml`:

```yaml
experiments:
  my_rag_experiment:
    datasets: [docfinqa_test_only]
    models: [gemini-flash, gpt4o-mini]
    evaluation_mode: retrieval_augmented  # Required for RAG
    rag_config: default_rag  # Reference to RAG config
    concurrency: 5
    limit: 50
    judge_model: openrouter-qwq-32b-judge
```

Then run:
```bash
finbench run my_rag_experiment
```

## Architecture

### Components

```
src/fin_bench/rag/
├── __init__.py           # Module exports
├── rag_config.py         # Configuration dataclass
├── embeddings.py         # Embedding model wrapper
├── chunking.py           # Text chunking strategies
├── vector_store.py       # FAISS vector store
├── retriever.py          # Retrieval + reranking
├── cache.py              # Index caching
└── pipeline.py           # Main RAG orchestrator
```

### Pipeline Flow

```
1. INDEXING (once per context)
   Context → Chunking → Embedding → Vector Store → Cache

2. RETRIEVAL (per question)
   Question → Embedding → Similarity Search → Reranking → Retrieved Context

3. GENERATION
   Retrieved Context + Question → LLM → Answer
```

## Configuration Options

### Core Settings

- **`enabled`**: Enable/disable RAG (default: `false`)
- **`embedding_model`**: HuggingFace model name
  - `BAAI/bge-large-en-v1.5`: Best quality (1024 dims)
  - `BAAI/bge-small-en-v1.5`: Fast (384 dims)
  - `sentence-transformers/all-mpnet-base-v2`: Good balance (768 dims)

### Chunking

- **`chunk_size`**: Target chunk size in tokens (default: `512`)
- **`chunk_overlap`**: Overlap between chunks (default: `50`)
- **`use_semantic_chunking`**: Use sentence boundaries (default: `false`)

### Retrieval

- **`top_k`**: Initial retrieval count (default: `20`)
- **`top_k_rerank`**: Final count after reranking (default: `5`)
- **`use_reranker`**: Enable cross-encoder reranking (default: `true`)
- **`reranker_model`**: Cross-encoder model name

### Vector Store

- **`vector_store_type`**: `"faiss"` (default)
- **`faiss_index_type`**: `"IndexFlatIP"` (exact search with inner product)
- **`use_gpu`**: Use GPU acceleration (default: `false`)

### Caching

- **`cache_dir`**: Cache directory (default: `".rag_cache"`)
- **`cache_enabled`**: Enable caching (default: `true`)

### Debug

- **`debug`**: Enable debug logging (default: `false`)
- **`print_chunks`**: Print chunks when indexing (default: `false`)
- **`print_retrieval`**: Print retrieval results (default: `false`)
- **`log_retrieval_stats`**: Log retrieval timing (default: `true`)

## Performance Tips

### 1. Use Caching
Caching is enabled by default. The first run will be slower as it builds indices, but subsequent runs will be much faster.

### 2. Adjust Chunk Size
- Larger chunks (512-1024 tokens): Better for capturing context
- Smaller chunks (128-256 tokens): More precise retrieval

### 3. Enable GPU
If you have a GPU:
```bash
pip install faiss-gpu
```

Then set `use_gpu: true` in your RAG config.

### 4. Tune top_k
- Higher `top_k`: More context, higher recall, slower
- Lower `top_k`: Less context, faster, might miss relevant info

### 5. Reranking Trade-off
- With reranking: Better quality, slower
- Without reranking: Faster, slightly lower quality

## Debugging

### Enable Debug Mode

Use `debug_rag` configuration or set debug flags:

```yaml
my_debug_rag:
  enabled: true
  debug: true
  print_chunks: true
  print_retrieval: true
```

### Check Cache

View cache statistics:
```python
from fin_bench.rag import RAGPipeline, RAGConfig

config = RAGConfig(enabled=True, cache_dir=".rag_cache")
pipeline = RAGPipeline(config)
pipeline.cache.print_stats()
```

### Clear Cache

```bash
rm -rf .rag_cache
```

Or programmatically:
```python
pipeline.clear_cache()
```

## Comparison: RAG vs Full Context

| Aspect | Full Context | RAG Pipeline |
|--------|-------------|-------------|
| **Context length** | Full document | Top 5-10 relevant chunks |
| **Token usage** | ~100k tokens | ~2k-5k tokens |
| **Latency** | Slower (long context) | Faster (smaller context) |
| **Cost** | Higher | Lower |
| **Accuracy** | May miss details in long docs | Focused on relevant info |
| **When to use** | Short docs, need all context | Long docs, targeted questions |

## Troubleshooting

### ImportError: sentence-transformers not found
```bash
pip install sentence-transformers
```

### ImportError: faiss not found
```bash
pip install faiss-cpu
# Or for GPU:
pip install faiss-gpu
```

### Slow embedding generation
- Use a smaller embedding model (`bge-small`)
- Increase `embedding_batch_size`
- Enable GPU acceleration

### Poor retrieval quality
- Increase `chunk_size` and `chunk_overlap`
- Enable `use_reranker`
- Try `use_semantic_chunking: true`
- Increase `top_k` and `top_k_rerank`

### Out of memory
- Decrease `embedding_batch_size`
- Use a smaller embedding model
- Process fewer samples at once (lower concurrency)

## Advanced Usage

### Custom Chunking Strategy

Implement a custom chunker in `src/fin_bench/rag/chunking.py`:

```python
class CustomChunker(TextChunker):
    def chunk(self, text: str, context_id: Optional[str] = None) -> List[Chunk]:
        # Your custom chunking logic
        pass
```

### Custom Embedding Model

Use any HuggingFace model:
```yaml
embedding_model: "your-org/your-model"
```

### Custom Reranker

Use any cross-encoder model:
```yaml
reranker_model: "your-org/your-cross-encoder"
```

## Citation

If you use this RAG implementation in your research, please cite:

```bibtex
@software{finbench_rag,
  title={RAG Pipeline for Financial Question Answering},
  author={FinBench Contributors},
  year={2025},
  url={https://github.com/your-org/finbench}
}
```

## Support

For issues or questions:
1. Check this README
2. Run the test script: `python test_rag_pipeline.py`
3. Enable debug mode to see detailed logs
4. Open an issue on GitHub
