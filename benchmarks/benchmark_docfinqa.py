import os
import json
import math
import textwrap
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Optional OpenAI import guarded to allow dry runs without the key
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


CHUNK_SIZE = 2750
CHUNK_OVERLAP = 550  # 20%
TOP_K = 5


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if size <= 0:
        raise ValueError("size must be > 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and < size")
    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        chunks.append(text[i : i + size])
        if i + size >= n:
            break
        i += size - overlap
    return chunks


def _to_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if s.endswith('%'):
        try:
            return float(s[:-1]) / 100.0
        except Exception:
            return None
    # Remove commas and currency
    s = re.sub(r"[,$]", "", s)
    try:
        return float(s)
    except Exception:
        return None


def numeric_equal(a: Any, b: Any, rel_tol: float = 5e-3, abs_tol: float = 1e-4) -> bool:
    fa = _to_float(a)
    fb = _to_float(b)
    if fa is None or fb is None:
        return False
    return math.isclose(fa, fb, rel_tol=rel_tol, abs_tol=abs_tol)


def safe_exec_python(code: str) -> Tuple[bool, Optional[Any], Optional[str]]:
    """
    Execute Python code in a restricted namespace. The code should assign a variable `answer`.
    Returns: (ok, answer, error)
    """
    safe_builtins = {
        'abs': abs,
        'min': min,
        'max': max,
        'sum': sum,
        'len': len,
        'range': range,
        'round': round,
        'float': float,
        'int': int,
        'str': str,
        'print': print,
    }
    globals_dict: Dict[str, Any] = {'__builtins__': safe_builtins}
    locals_dict: Dict[str, Any] = {}
    try:
        exec(code, globals_dict, locals_dict)
    except Exception as e:
        return False, None, f"exec_error: {e}"
    if 'answer' not in locals_dict:
        return False, None, "answer_not_found"
    return True, locals_dict['answer'], None


def make_few_shots() -> str:
    example = {
        "Context": "For the quarter December 31, 2012, company repurchased 619,314 shares; December purchases: 102,400.",
        "Question": "For the quarter Dec 31, 2012 what percent of total shares were purchased in December?",
        "Program": """
dec_2012 = 102_400
q4_total = 619_314
answer = dec_2012 / q4_total
""".strip(),
        "Answer": "0.165"  # 16.5%
    }
    return textwrap.dedent(f"""
    You are given a Context, a Question, and must write Python code that computes the numeric answer.
    - Use readable variable names grounded in the context.
    - Assign the final numeric result to a variable named `answer`.

    Example:
    Context:
    {example['Context']}
    Question: {example['Question']}
    Python:
    ```python
    {example['Program']}
    ```
    Answer: {example['Answer']}
    """)


def build_prompt(chunks: List[str], question: str) -> str:
    header = make_few_shots()
    joined_chunks = "\n\n".join([f"[Chunk {i+1}]\n{c}" for i, c in enumerate(chunks)])
    return textwrap.dedent(f"""
    {header}

    Use the following retrieved context chunks to answer the query by writing Python code.
    Retrieved Chunks:
    {joined_chunks}

    Query: {question}
    Return only a Python code block that assigns `answer`.
    """)


def extract_code(text: str) -> str:
    code_blocks = re.findall(r"```python\n([\s\S]*?)```", text)
    if code_blocks:
        return code_blocks[0]
    # fallback: entire text
    return text.strip()


@dataclass
class RetrievalIndex:
    model_name: str
    encoder: SentenceTransformer


def build_index(chunks: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> RetrievalIndex:
    encoder = SentenceTransformer(model_name)
    return RetrievalIndex(model_name=model_name, encoder=encoder)


def retrieve_top_k(index: RetrievalIndex, chunks: List[str], question: str, k: int = TOP_K) -> List[str]:
    enc = index.encoder
    q_emb = enc.encode([question], show_progress_bar=False, convert_to_numpy=True)
    c_emb = enc.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    sims = cosine_similarity(q_emb, c_emb)[0]
    top_idx = np.argsort(-sims)[:k]
    return [chunks[i] for i in top_idx]


def llm_program(chunks: List[str], question: str, model: str = "gpt-4o-mini") -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        raise RuntimeError("OpenAI not configured. Set OPENAI_API_KEY or install openai>=1.40.0")
    client = OpenAI(api_key=api_key)
    prompt = build_prompt(chunks, question)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You write short, correct Python programs that compute numeric answers."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=400,
    )
    text = resp.choices[0].message.content or ""
    return extract_code(text)


def run_benchmark(split: str = "test", k: int = TOP_K, use_llm: bool = True, limit: Optional[int] = None) -> Dict[str, Any]:
    data_path = os.path.join(os.path.dirname(__file__), "DocFinQA", f"{split}.json")
    with open(data_path, "r") as f:
        data = json.load(f)

    total = len(data)
    if limit is not None:
        data = data[:limit]

    correct = 0
    results = []

    for ex in tqdm(data, desc=f"Benchmark {split}"):
        context = ex.get("Context", "")
        question = ex.get("Question", "")
        gold_program = ex.get("Program", "")
        gold_answer = ex.get("Answer", None)

        chunks = chunk_text(context)
        if not chunks:
            results.append({"ok": False, "reason": "no_chunks"})
            continue

        index = build_index(chunks)
        top_chunks = retrieve_top_k(index, chunks, question, k=k)

        if use_llm:
            try:
                prog = llm_program(top_chunks, question)
            except Exception as e:
                prog = ""
                llm_error = str(e)
        else:
            prog = gold_program
            llm_error = None

        ok, pred, err = safe_exec_python(prog) if prog else (False, None, "no_program")
        if not ok and not use_llm:
            # Last resort: try gold program if provided
            ok, pred, err = safe_exec_python(gold_program)

        is_correct = numeric_equal(pred, gold_answer)
        correct += 1 if is_correct else 0
        results.append({
            "question": question,
            "pred": pred,
            "gold": gold_answer,
            "correct": bool(is_correct),
            "exec_error": err,
            "llm_error": llm_error if use_llm else None,
        })

    acc = correct / (len(data) if data else 1)
    summary = {"split": split, "evaluated": len(data), "total_in_file": total, "accuracy": acc}
    print(summary)
    return {"summary": summary, "results": results}


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"]) 
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_llm", action="store_true", help="use gold program instead of LLM")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    run_benchmark(split=args.split, k=args.top_k, use_llm=(not args.no_llm), limit=args.limit)


