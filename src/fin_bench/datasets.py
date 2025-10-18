"""
Dataset loaders for all supported financial QA benchmarks.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Iterator, Optional
from dataclasses import dataclass

from .types import Dataset, DatasetType, SampleDict
from .config import ConfigError

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False


@dataclass
class DatasetSample:
    """Standardized dataset sample."""
    id: str
    question: str
    context: Optional[str] = None
    program: Optional[str] = None
    answer: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DatasetLoader:
    """Base class for dataset loaders."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self._sample_cache: Dict[str, List[DatasetSample]] = {}

    def load_split(self, split: str) -> List[DatasetSample]:
        """Load samples for a specific split."""
        if split in self._sample_cache:
            return self._sample_cache[split]

        samples = self._load_split_impl(split)
        self._sample_cache[split] = samples
        return samples

    def _load_split_impl(self, split: str) -> List[DatasetSample]:
        """Implementation-specific loading logic."""
        raise NotImplementedError

    def get_sample_fields(self) -> List[str]:
        """Get available fields in samples."""
        return self.dataset.sample_fields

    def validate_sample(self, sample: DatasetSample) -> bool:
        """Validate that a sample has required fields."""
        required_fields = ['id', 'question']
        for field in required_fields:
            if not hasattr(sample, field) or getattr(sample, field) is None:
                return False
        return True


class DocFinQALoader(DatasetLoader):
    """Loader for DocFinQA dataset."""

    def _load_split_impl(self, split: str) -> List[DatasetSample]:
        """Load DocFinQA split using HuggingFace datasets."""
        if not HF_DATASETS_AVAILABLE:
            raise ConfigError("HuggingFace datasets library is required. Install with: pip install datasets")

        split_path = self.dataset.path / f"{split}.json"
        if not split_path.exists():
            raise ConfigError(f"DocFinQA split not found: {split_path}")

        # Load using HuggingFace datasets - handles both JSON and JSONL automatically
        try:
            hf_dataset = load_dataset('json', data_files=str(split_path), split='train')
            data = list(hf_dataset)
        except Exception as e:
            raise ConfigError(f"Failed to load DocFinQA split {split}: {e}")

        samples = []
        for obj in data:
            sample = self._parse_docfinqa_sample(obj)
            if sample:
                samples.append(sample)

        return samples

    def _parse_docfinqa_sample(self, obj: Dict[str, Any]) -> Optional[DatasetSample]:
        """Parse a DocFinQA sample."""
        try:
            sample_id = obj.get('id', f"docfinqa_{len(self._sample_cache.get('current', []))}")
            question = obj.get('Question', '')
            context = obj.get('Context', '')
            program = obj.get('Program', '')
            answer = obj.get('Answer', '')

            return DatasetSample(
                id=sample_id,
                question=question,
                context=context,
                program=program,
                answer=answer,
                metadata={'dataset': 'docfinqa', 'has_program': bool(program)}
            )
        except Exception:
            return None


class FinQALoader(DatasetLoader):
    """Loader for FinQA dataset."""

    def _load_split_impl(self, split: str) -> List[DatasetSample]:
        """Load FinQA split."""
        split_path = self.dataset.path / f"{split}.json"

        if not split_path.exists():
            raise ConfigError(f"FinQA split not found: {split_path}")

        samples = []
        with open(split_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for obj in data:
            sample = self._parse_finqa_sample(obj)
            if sample:
                samples.append(sample)

        return samples

    def _parse_finqa_sample(self, obj: Dict[str, Any]) -> Optional[DatasetSample]:
        """Parse a FinQA sample.
        
        FinQA format has:
        - pre_text: list of sentences
        - table/table_ori: table data
        - post_text: list of sentences  
        - qa: dict with question, answer, program, model_input (retrieved facts - IGNORE)
        
        For coding agents: Build FULL context from pre_text + table + post_text
        DO NOT use model_input (that's pre-retrieved facts from the paper's BERT retriever)
        """
        try:
            sample_id = obj.get('id', '')
            
            # Extract question, answer, program from qa field
            qa = obj.get('qa', {})
            if not qa:
                return None
                
            question = qa.get('question', '')
            answer = qa.get('answer', '')
            program = qa.get('program', '')
            
            # Build FULL context (no retrieval) from pre_text + table + post_text
            sections: List[str] = []
            
            # Pre-text
            pre_text_chunks = obj.get('pre_text', [])
            if pre_text_chunks:
                pre_text = '\n\n'.join(str(x) for x in pre_text_chunks if x)
                if pre_text.strip():
                    sections.append(pre_text)
            
            # Table as Markdown
            table_rows = obj.get('table_ori') or obj.get('table') or []
            if isinstance(table_rows, list) and table_rows:
                # Normalize row lengths
                max_len = max((len(r) for r in table_rows if isinstance(r, list)), default=0)
                norm_rows: List[List[str]] = []
                for r in table_rows:
                    if isinstance(r, list):
                        cells = [str(c) if c is not None else '' for c in r]
                        if len(cells) < max_len:
                            cells = cells + [''] * (max_len - len(cells))
                        norm_rows.append(cells)
                
                if norm_rows:
                    header = norm_rows[0]
                    divider = ['-' * max(3, len(h)) for h in header]
                    lines = [
                        '| ' + ' | '.join(header) + ' |',
                        '| ' + ' | '.join(divider) + ' |'
                    ]
                    for r in norm_rows[1:]:
                        lines.append('| ' + ' | '.join(r) + ' |')
                    sections.append('\n'.join(lines))
            
            # Post-text
            post_text_chunks = obj.get('post_text', [])
            if post_text_chunks:
                post_text = '\n\n'.join(str(x) for x in post_text_chunks if x)
                if post_text.strip():
                    sections.append(post_text)
            
            context = '\n\n'.join(sections)

            return DatasetSample(
                id=sample_id,
                question=question,
                context=context,
                program=program,
                answer=str(answer) if answer is not None else '',
                metadata={
                    'dataset': 'finqa',
                    'has_table': bool(table_rows),
                    # NOTE: qa['model_input'] contains pre-retrieved facts from BERT retriever
                    # We intentionally DO NOT use it - agents get full context
                }
            )
        except Exception:
            return None


class TATQALoader(DatasetLoader):
    """Loader for TAT-QA dataset."""

    def _load_split_impl(self, split: str) -> List[DatasetSample]:
        """Load TAT-QA split.
        
        TAT-QA structure: Each item has one table+paragraphs but MULTIPLE questions.
        We create one DatasetSample per question.
        
        For test split: Use tatqa_dataset_test_gold.json (has answers for evaluation)
        """
        # For test split, use the gold file which has answers
        if split == "test":
            split_path = self.dataset.path / "tatqa_dataset_test_gold.json"
        else:
            split_path = self.dataset.path / f"tatqa_dataset_{split}.json"

        if not split_path.exists():
            raise ConfigError(f"TAT-QA split not found: {split_path}")

        samples = []
        with open(split_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # TAT-QA format: list of contexts, each with table+paragraphs+multiple questions
        if isinstance(data, list):
            for context_idx, context_obj in enumerate(data):
                # Each context has multiple questions
                questions = context_obj.get('questions', [])
                for q in questions:
                    sample = self._parse_tatqa_sample(context_idx, context_obj, q, all_questions=questions)
                    if sample:
                        samples.append(sample)
        else:
            # Fallback for old format
            for qid, obj in data.items():
                sample = self._parse_tatqa_sample_old(qid, obj)
                if sample:
                    samples.append(sample)

        return samples

    def _parse_tatqa_sample(self, context_idx: int, context_obj: Dict[str, Any], question_obj: Dict[str, Any], all_questions: List[Dict[str, Any]] = None) -> Optional[DatasetSample]:
        """Parse a TAT-QA sample from new format.
        
        TAT-QA structure:
        - context_obj: {'table': {...}, 'paragraphs': [...]}
        - question_obj: {'uid': ..., 'question': ..., 'answer': ..., 'derivation': ..., 'answer_type': ...}
        - all_questions: List of all question objects from this context (for include_all_questions mode)
        """
        try:
            # Extract question details
            question_id = question_obj.get('uid', f"tatqa_{context_idx}_{question_obj.get('order', 0)}")
            question = question_obj.get('question', '')
            answer = question_obj.get('answer', '')
            answer_type = question_obj.get('answer_type', 'unknown')
            
            # Build context: Table + Paragraphs (hybrid context as per TAT-QA paper)
            sections = []
            
            # 1. Extract and convert table to Markdown
            table_obj = context_obj.get('table', {})
            if isinstance(table_obj, dict):
                table_data = table_obj.get('table', [])
            else:
                table_data = table_obj
                
            if isinstance(table_data, list) and table_data:
                # Normalize row lengths
                max_len = max((len(r) for r in table_data if isinstance(r, list)), default=0)
                norm_rows: List[List[str]] = []
                for r in table_data:
                    if isinstance(r, list):
                        cells = [str(c) if c is not None else '' for c in r]
                        if len(cells) < max_len:
                            cells = cells + [''] * (max_len - len(cells))
                        norm_rows.append(cells)
                
                if norm_rows:
                    header = norm_rows[0] if norm_rows else []
                    divider = ['-' * max(3, len(h)) for h in header]
                    lines = [
                        '| ' + ' | '.join(header) + ' |',
                        '| ' + ' | '.join(divider) + ' |'
                    ]
                    for r in norm_rows[1:]:
                        lines.append('| ' + ' | '.join(r) + ' |')
                    sections.append("## Table\n\n" + '\n'.join(lines))
            
            # 2. Extract paragraphs (the "nearby explanatory paragraphs")
            paragraphs = context_obj.get('paragraphs', [])
            if isinstance(paragraphs, list) and paragraphs:
                # Each paragraph is a dict with 'text' field
                para_texts = []
                for p in paragraphs:
                    if isinstance(p, dict):
                        para_texts.append(p.get('text', ''))
                    else:
                        para_texts.append(str(p))
                
                text_block = '\n\n'.join(t for t in para_texts if t.strip())
                if text_block.strip():
                    sections.append("## Text\n\n" + text_block)
            
            context = "\n\n".join(sections)
            
            # Handle answer format (could be list or string)
            if isinstance(answer, list):
                answer = ', '.join(str(a) for a in answer)
            else:
                answer = str(answer) if answer else ''

            # Store all questions from context if provided (for include_all_questions mode)
            metadata = {
                'dataset': 'tatqa',
                'has_table': bool(table_data),
                'answer_type': answer_type,
                'derivation': question_obj.get('derivation', ''),
                'scale': question_obj.get('scale', ''),
                'include_all_questions': self.dataset.include_all_questions  # Pass dataset config flag
            }
            
            # Add all questions from context to metadata (for include_all_questions mode)
            if all_questions:
                metadata['all_context_questions'] = [
                    {
                        'order': q.get('order'),
                        'question': q.get('question', ''),
                        'uid': q.get('uid', '')
                    }
                    for q in all_questions
                ]
            
            return DatasetSample(
                id=question_id,
                question=question,
                context=context,
                answer=answer,
                metadata=metadata
            )
        except Exception as e:
            print(f"Error parsing TAT-QA sample {context_idx}: {e}")
            return None
    
    def _parse_tatqa_sample_old(self, qid: str, obj: Dict[str, Any]) -> Optional[DatasetSample]:
        """Parse old TAT-QA format (fallback)."""
        try:
            question = obj.get('question', '')
            context = obj.get('context', '')
            answer = obj.get('answer', '')
            
            return DatasetSample(
                id=qid,
                question=question,
                context=context,
                answer=answer,
                metadata={
                    'dataset': 'tatqa',
                    'has_table': False,
                    'answer_type': obj.get('answer_type', 'unknown')
                }
            )
        except Exception:
            return None


class FinanceBenchLoader(DatasetLoader):
    """Loader for FinanceBench dataset."""

    def _load_split_impl(self, split: str) -> List[DatasetSample]:
        """Load FinanceBench split."""
        # FinanceBench uses JSONL format
        split_path = self.dataset.path / "financebench_open_source.jsonl"

        if not split_path.exists():
            raise ConfigError(f"FinanceBench data not found: {split_path}")

        samples = []
        with open(split_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    obj = json.loads(line.strip())
                    sample = self._parse_financebench_sample(obj)
                    if sample:
                        samples.append(sample)
                except json.JSONDecodeError:
                    continue

        return samples

    def _parse_financebench_sample(self, obj: Dict[str, Any]) -> Optional[DatasetSample]:
        """Parse a FinanceBench sample.
        
        For coding agents: Use evidence pages directly (Oracle setup from paper).
        NO retrieval, NO vector stores - just the evidence page text.
        """
        try:
            sample_id = obj.get('financebench_id', f"fb_{len(self._sample_cache.get('current', []))}")
            question = obj.get('question', '')
            answer = obj.get('answer', '')
            
            # Build context from evidence (Oracle setup - evidence pages)
            # Evidence is a list of dicts with evidence_text_full_page
            context = ""
            evidence_list = obj.get('evidence', [])
            
            if isinstance(evidence_list, list) and evidence_list:
                # Extract full evidence pages
                evidence_pages = []
                for ev in evidence_list:
                    if isinstance(ev, dict):
                        # Use evidence_text_full_page (complete page)
                        full_page = ev.get('evidence_text_full_page', '')
                        if not full_page:
                            # Fallback to evidence_text (excerpt)
                            full_page = ev.get('evidence_text', '')
                        
                        if full_page:
                            doc_name = ev.get('doc_name', 'Unknown')
                            page_num = ev.get('evidence_page_num', 'N/A')
                            evidence_pages.append(
                                f"## Evidence Page (Document: {doc_name}, Page: {page_num})\n\n{full_page}"
                            )
                
                context = "\n\n".join(evidence_pages)
            
            # If no evidence, try context field (fallback)
            if not context:
                context = obj.get('context', '')

            return DatasetSample(
                id=sample_id,
                question=question,
                context=context,
                answer=answer,
                metadata={
                    'dataset': 'financebench',
                    'company': obj.get('company', ''),
                    'doc_name': obj.get('doc_name', ''),
                    'question_type': obj.get('question_type', ''),
                    'answer_type': obj.get('question_type', 'unknown'),
                    'justification': obj.get('justification', ''),
                    'num_evidence_pages': len(evidence_list) if isinstance(evidence_list, list) else 0
                }
            )
        except Exception as e:
            print(f"Error parsing FinanceBench sample: {e}")
            return None


class EconLogicQALoader(DatasetLoader):
    """Loader for EconLogicQA dataset."""

    def _load_split_impl(self, split: str) -> List[DatasetSample]:
        """Load EconLogicQA split."""
        # Handle nested structure
        split_dir = self.dataset.path / "econlogicqa.json" / f"{split}.json"

        if not split_dir.exists():
            # Try direct path
            split_dir = self.dataset.path / f"{split}.json"
            if not split_dir.exists():
                raise ConfigError(f"EconLogicQA split not found: {split_dir}")

        samples = []
        with open(split_dir, 'r', encoding='utf-8') as f:
            # Handle JSON Lines format
            for line_num, line in enumerate(f):
                try:
                    obj = json.loads(line.strip())
                    sample = self._parse_econlogicqa_sample(f"sample_{line_num}", obj)
                    if sample:
                        samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"  ⚠️  Skipping invalid JSON line {line_num + 1}: {e}")
                    continue

        return samples

    def _parse_econlogicqa_sample(self, sample_id: str, obj: Dict[str, Any]) -> Optional[DatasetSample]:
        """Parse an EconLogicQA sample.
        
        EconLogicQA format:
        - Question: Scenario description + instruction to order events
        - A, B, C, D: The four events to be ordered logically
        - Answer: Correct order (e.g., "D, A, C, B")
        
        For coding agents: Provide question + events, agent must order them.
        NO news article context (that was only for dataset generation).
        """
        try:
            # Question contains the scenario + instruction
            question = obj.get('Question', '')
            answer = obj.get('Answer', '')
            
            # Extract the four events (A, B, C, D)
            options = []
            for letter in ['A', 'B', 'C', 'D']:
                event = obj.get(letter, '')
                if event:
                    options.append(f"{letter}) {event}")
            
            # Context is empty - no news article needed for evaluation
            # (News articles were only used to GENERATE the questions)
            context = ''

            return DatasetSample(
                id=sample_id,
                question=question,
                context=context,
                answer=answer,
                metadata={
                    'dataset': 'econlogicqa',
                    'options': options,
                    'explanation': obj.get('explanation', ''),
                    'answer_type': 'multiple_choice'
                }
            )
        except Exception as e:
            print(f"Error parsing EconLogicQA sample {sample_id}: {e}")
            return None


class BizBenchLoader(DatasetLoader):
    """Loader for BizBench-QA dataset."""

    def _load_split_impl(self, split: str) -> List[DatasetSample]:
        """Load BizBench split."""
        # Handle nested structure
        split_dir = self.dataset.path / "bizbench.json" / f"{split}.json"

        if not split_dir.exists():
            # Try direct path
            split_dir = self.dataset.path / f"{split}.json"
            if not split_dir.exists():
                raise ConfigError(f"BizBench split not found: {split_dir}")

        samples = []
        with open(split_dir, 'r', encoding='utf-8') as f:
            # Handle JSON Lines format
            for line_num, line in enumerate(f):
                try:
                    obj = json.loads(line.strip())
                    sample = self._parse_bizbench_sample(f"sample_{line_num}", obj)
                    if sample:
                        samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"  ⚠️  Skipping invalid JSON line {line_num + 1}: {e}")
                    continue

        return samples

    def _parse_bizbench_sample(self, sample_id: str, obj: Dict[str, Any]) -> Optional[DatasetSample]:
        """Parse a BizBench sample.
        
        BizBench contains 8 different tasks:
        1. FinCode - Question only (CFA/CPA problems)
        2. CodeFinQA - Question + text/tables
        3. CodeTAT-QA - Question + structured table
        4. SEC-Num - Document snippet + target label
        5. ConvFinQA (Extract) - Question + text/tables
        6. TAT-QA (Extract) - Question + table/text
        7. FinKnow - Multiple choice (domain knowledge)
        8. FormulaEval - Function stub + docstring
        
        Context can be empty for knowledge tasks (FinCode, FinKnow, FormulaEval).
        """
        try:
            question = obj.get('question', '')
            answer = obj.get('answer', '')
            
            # Context can be None for some tasks
            context = obj.get('context', '')
            if context is None:
                context = ''
            
            # Task type identifies which of the 8 tasks
            task = obj.get('task', 'unknown')  # Field is 'task' not 'task_type'

            return DatasetSample(
                id=sample_id,
                question=question,
                context=context,
                answer=answer,
                metadata={
                    'dataset': 'bizbench',
                    'task_type': task,
                    'context_type': obj.get('context_type'),
                    'has_options': obj.get('options') is not None,
                    'has_program': obj.get('program') is not None,
                    'domain': obj.get('domain', 'general'),
                    'difficulty': obj.get('difficulty', 'unknown')
                }
            )
        except Exception:
            return None


class ConvFinQALoader(DatasetLoader):
    """Loader for ConvFinQA dataset."""

    def _load_split_impl(self, split: str) -> List[DatasetSample]:
        """Load ConvFinQA split."""
        # ConvFinQA has data in data/ subdirectory with special naming
        if split == "test":
            split_path = self.dataset.path / "data" / "test_private.json"
        else:
            split_path = self.dataset.path / "data" / f"{split}.json"

        if not split_path.exists():
            # Try direct path
            if split == "test":
                # Try multiple common variants
                candidates = [
                    self.dataset.path / "test_private.json",
                    self.dataset.path / "data" / "test.json",
                    self.dataset.path / "test.json",
                ]
                for cand in candidates:
                    if cand.exists():
                        split_path = cand
                        break
                else:
                    raise ConfigError(f"ConvFinQA split not found: {split_path}")
            else:
                split_path = self.dataset.path / f"{split}.json"
                if not split_path.exists():
                    raise ConfigError(f"ConvFinQA split not found: {split_path}")

        samples = []
        with open(split_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for i, obj in enumerate(data):
            sample = self._parse_convfinqa_sample(f"sample_{i}", obj)
            if sample:
                samples.append(sample)

        return samples

    def _parse_convfinqa_sample(self, sample_id: str, obj: Dict[str, Any]) -> Optional[DatasetSample]:
        """Parse a ConvFinQA sample."""
        try:
            # ConvFinQA format: pre_text, post_text, qa field
            qa = obj.get('qa', {})
            if not qa:
                return None

            question = qa.get('question', '')

            # Build rich context: pre_text + tables + post_text
            pre_text_chunks = obj.get('pre_text', [])
            post_text_chunks = obj.get('post_text', [])
            table_rows = obj.get('table_ori') or obj.get('table') or []

            def join_text_chunks(chunks: Any) -> str:
                if isinstance(chunks, list):
                    return '\n'.join(str(x) for x in chunks if x is not None)
                return str(chunks)

            def convert_table_to_markdown(rows: Any) -> str:
                if not isinstance(rows, list) or not rows:
                    return ""
                # If rows is a list of lists, treat as a single table; otherwise, return stringified
                if all(isinstance(r, list) for r in rows):
                    # Normalize row lengths
                    max_len = max((len(r) for r in rows if isinstance(r, list)), default=0)
                    norm_rows: List[List[str]] = []
                    for r in rows:
                        if isinstance(r, list):
                            cells = [str(c) if c is not None else '' for c in r]
                            if len(cells) < max_len:
                                cells = cells + [''] * (max_len - len(cells))
                            norm_rows.append(cells)
                    if not norm_rows:
                        return ""
                    header = norm_rows[0]
                    divider = ['-' * max(3, len(h)) for h in header]
                    lines = [
                        '| ' + ' | '.join(header) + ' |',
                        '| ' + ' | '.join(divider) + ' |'
                    ]
                    for r in norm_rows[1:]:
                        lines.append('| ' + ' | '.join(r) + ' |')
                    return '\n'.join(lines)
                # If rows is nested (list of tables), render each table recursively
                if all(isinstance(r, list) for r in rows):
                    return convert_table_to_markdown(rows)
                return str(rows)

            sections: List[str] = []

            pre_text = join_text_chunks(pre_text_chunks)
            if pre_text.strip():
                sections.append("## Context (Pre Text)\n\n" + pre_text)

            table_md = convert_table_to_markdown(table_rows)
            if table_md.strip():
                sections.append("## Context (Table)\n\n" + table_md)

            post_text = join_text_chunks(post_text_chunks)
            if post_text.strip():
                sections.append("## Context (Post Text)\n\n" + post_text)

            context_text = "\n\n".join(sections)

            # Try to get answer from qa
            answer = qa.get('answer', '')
            
            # Extract conversation history from annotation if available
            annotation = obj.get('annotation', {})
            conversation_history = None
            if annotation and 'dialogue_break' in annotation:
                # dialogue_break contains the individual turns of the conversation
                dialogue_turns = annotation['dialogue_break']
                answer_list = annotation.get('answer_list', [])
                
                if isinstance(dialogue_turns, list) and len(dialogue_turns) > 1:
                    # Build conversation history: list of (question, answer) pairs
                    conversation_history = []
                    for i, turn_question in enumerate(dialogue_turns):
                        turn_answer = answer_list[i] if i < len(answer_list) else ''
                        conversation_history.append({
                            'question': turn_question,
                            'answer': turn_answer
                        })

            metadata = {
                'dataset': 'convfinqa',
                'conversational': True,
                'has_conversation_history': conversation_history is not None,
            }
            
            if conversation_history:
                metadata['conversation_history'] = conversation_history
                metadata['num_turns'] = len(conversation_history)

            return DatasetSample(
                id=obj.get('id', sample_id),  # Use original ID if available
                question=question,
                context=context_text,
                program=qa.get('program', ''),
                answer=answer,
                metadata=metadata
            )
        except Exception:
            return None


class DocMathEvalLoader(DatasetLoader):
    """Loader for DocMath-Eval dataset."""

    def _load_split_impl(self, split: str) -> List[DatasetSample]:
        """Load DocMath-Eval split."""
        # Handle nested structure
        split_dir = self.dataset.path / "docmath_eval.json" / f"{split}.json"

        if not split_dir.exists():
            # Try direct path
            split_dir = self.dataset.path / f"{split}.json"
            if not split_dir.exists():
                print(f"⚠️  DocMath-Eval split not found: {split_dir}")
                return []

        samples = []
        with open(split_dir, 'r', encoding='utf-8') as f:
            # Handle JSON Lines format
            for line_num, line in enumerate(f):
                try:
                    obj = json.loads(line.strip())
                    sample = self._parse_docmath_eval_sample(f"sample_{line_num}", obj)
                    if sample:
                        samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"  ⚠️  Skipping invalid JSON line {line_num + 1}: {e}")
                    continue

        return samples

    def _parse_docmath_eval_sample(self, sample_id: str, obj: Dict[str, Any]) -> Optional[DatasetSample]:
        """Parse a DocMath-Eval sample."""
        try:
            # DocMath-Eval format: question, python_solution, ground_truth
            question = obj.get('question', '')
            context = obj.get('paragraphs', '')
            program = obj.get('python_solution', '')
            answer = obj.get('ground_truth', '')

            # Combine paragraphs into context
            if isinstance(context, list):
                context_text = '\n\n'.join(context)
            else:
                context_text = str(context)

            return DatasetSample(
                id=sample_id,
                question=question,
                context=context_text,
                program=program,
                answer=answer,
                metadata={
                    'dataset': 'docmath_eval',
                    'question_id': obj.get('question_id', ''),
                    'source': obj.get('source', ''),
                    'table_evidence': obj.get('table_evidence', []),
                    'paragraph_evidence': obj.get('paragraph_evidence', [])
                }
            )
        except Exception:
            return None


class FINER139Loader(DatasetLoader):
    """Loader for FINER-139 dataset from local JSONL files."""

    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        # Load tag names once and cache them
        self.tag_names = None
        self.tag_to_index = None

    def _load_tag_vocabulary(self):
        """Load and build tag vocabulary from all splits."""
        if self.tag_names is not None:
            return self.tag_names, self.tag_to_index

        # Collect all unique tags from all splits
        all_tags = set()

        for split in ['train', 'validation', 'test']:
            split_path = self.dataset.path / f"{split}.jsonl"
            if not split_path.exists():
                continue

            with open(split_path, 'r', encoding='utf-8') as f:
                # Sample first 1000 lines to build vocabulary (faster than full scan)
                for i, line in enumerate(f):
                    if i >= 1000:  # Sample size for vocabulary building
                        break
                    try:
                        obj = json.loads(line.strip())
                        tags = obj.get('ner_tags', [])
                        all_tags.update(tags)
                    except json.JSONDecodeError:
                        continue

        # Sort tags: 'O' first, then alphabetically
        sorted_tags = sorted(all_tags, key=lambda x: (x != 'O', x))

        # Build mappings
        self.tag_names = sorted_tags
        self.tag_to_index = {tag: idx for idx, tag in enumerate(sorted_tags)}

        return self.tag_names, self.tag_to_index

    def _load_split_impl(self, split: str) -> List[DatasetSample]:
        """Load FINER-139 split from local JSONL file."""
        split_path = self.dataset.path / f"{split}.jsonl"

        if not split_path.exists():
            raise ConfigError(f"FINER-139 split not found: {split_path}")

        # Load tag vocabulary
        tag_names, tag_to_index = self._load_tag_vocabulary()

        samples = []
        with open(split_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    obj = json.loads(line.strip())
                    sample = self._parse_finer139_sample(idx, obj, tag_names, tag_to_index)
                    if sample:
                        samples.append(sample)
                except json.JSONDecodeError:
                    print(f"  ⚠️  Skipping invalid JSON line {idx + 1}")
                    continue

        return samples

    def _parse_finer139_sample(self, idx: int, obj: Dict[str, Any], tag_names: List[str], tag_to_index: Dict[str, int]) -> Optional[DatasetSample]:
        """Parse a FINER-139 sample."""
        try:
            tokens = obj.get('tokens', [])
            ner_tag_strings = obj.get('ner_tags', [])

            if not tokens or not ner_tag_strings:
                return None

            # Convert string tags to integer indices
            ner_tag_indices = [tag_to_index.get(tag, 0) for tag in ner_tag_strings]

            # Build context with tag mapping and instructions
            context = self._build_context(tokens, tag_names)

            # Question
            question = "Tag each token with its corresponding XBRL entity type index."

            # Answer: store as JSON string of the integer tag indices
            answer = json.dumps(ner_tag_indices)

            return DatasetSample(
                id=f"finer139_{idx}",
                question=question,
                context=context,
                answer=answer,
                metadata={
                    'dataset': 'finer139',
                    'num_tokens': len(tokens),
                    'num_tags': len(tag_names)
                }
            )
        except Exception as e:
            print(f"Error parsing FINER-139 sample {idx}: {e}")
            return None

    def _build_context(self, tokens: List[str], tag_names: List[str]) -> str:
        """Build context string with tag mapping and instructions."""
        # Create tag mapping string (indices 0-278)
        tag_mapping_lines = [f"{i}: {name}" for i, name in enumerate(tag_names)]
        tag_mapping_str = "\n".join(tag_mapping_lines)

        # Build full context
        context = f"""## Task: XBRL Entity Tagging

You are given a list of tokens from a financial document.
Your task is to tag each token with its corresponding XBRL entity type index.

## Tag Mapping (indices 0-{len(tag_names)-1}):

```
{tag_mapping_str}
```

## Tokens to Tag:

```python
{json.dumps(tokens)}
```

## Instructions:

1. Analyze each token and determine if it represents an XBRL financial entity
2. If it's an entity, assign the appropriate index from the tag mapping above
3. If it's NOT an entity, use index 0 (tag "O")
4. Write your answer to `answer.py` as a Python list of integers:

```python
answer = [0, 1, 2, 0, 45, ...]  # One integer index per token
```

**Important Notes:**
- The tag mapping uses BIO format: B- (Begin entity), I- (Inside/continuation of entity), O (Outside/not an entity)
- Index 0 corresponds to "O" (not an entity)
- Indices 1-{len(tag_names)-1} correspond to specific XBRL entity tags
- Your answer list must have exactly {len(tokens)} integers
- Each integer must be in the range 0-{len(tag_names)-1}
"""

        return context


class DatasetFactory:
    """Factory for creating dataset loaders."""

    _loaders = {
        DatasetType.DOCFINQA: DocFinQALoader,
        DatasetType.FINQA: FinQALoader,
        DatasetType.CONVFINQA: ConvFinQALoader,
        DatasetType.TATQA: TATQALoader,
        DatasetType.FINANCEBENCH: FinanceBenchLoader,
        DatasetType.ECONLOGICQA: EconLogicQALoader,
        DatasetType.BIZBENCH: BizBenchLoader,
        DatasetType.DOCMATH_EVAL: DocMathEvalLoader,
        DatasetType.FINER139: FINER139Loader,
    }

    @classmethod
    def create_loader(cls, dataset: Dataset) -> DatasetLoader:
        """Create a loader for the given dataset type."""
        if dataset.type not in cls._loaders:
            raise ConfigError(f"No loader available for dataset type: {dataset.type}")

        loader_class = cls._loaders[dataset.type]
        return loader_class(dataset)

    @classmethod
    def get_supported_types(cls) -> List[DatasetType]:
        """Get list of supported dataset types."""
        return list(cls._loaders.keys())


def load_samples(dataset: Dataset, split: str, limit: Optional[int] = None) -> List[DatasetSample]:
    """Convenience function to load samples from a dataset."""
    loader = DatasetFactory.create_loader(dataset)
    samples = loader.load_split(split)

    if limit:
        samples = samples[:limit]

    return samples


def validate_dataset_structure(dataset: Dataset) -> bool:
    """Validate that dataset directory structure is correct."""
    # FINER139 loads from local JSONL files
    if dataset.type == DatasetType.FINER139:
        if not dataset.path.exists():
            return False
        # Check for at least one JSONL split file
        for split in dataset.splits:
            split_path = dataset.path / f"{split}.jsonl"
            if split_path.exists():
                return True
        return False

    if not dataset.path.exists():
        return False

    # Check if at least one split exists
    for split in dataset.splits:
        if dataset.type == DatasetType.DOCFINQA:
            split_path = dataset.path / f"{split}.json"
        elif dataset.type == DatasetType.FINQA:
            split_path = dataset.path / f"{split}.json"
        elif dataset.type == DatasetType.TATQA:
            split_path = dataset.path / f"tatqa_dataset_{split}.json"
        elif dataset.type == DatasetType.FINANCEBENCH:
            split_path = dataset.path / "financebench_open_source.jsonl"
        elif dataset.type == DatasetType.ECONLOGICQA:
            # Handle nested structure
            split_path = dataset.path / "econlogicqa.json" / f"{split}.json"
            if not split_path.exists():
                split_path = dataset.path / f"{split}.json"
        elif dataset.type == DatasetType.BIZBENCH:
            # Handle nested structure
            split_path = dataset.path / "bizbench.json" / f"{split}.json"
            if not split_path.exists():
                split_path = dataset.path / f"{split}.json"
        elif dataset.type == DatasetType.CONVFINQA:
            # ConvFinQA has data in data/ subdirectory with special naming
            if split == "test":
                split_path = dataset.path / "data" / "test_private.json"
            else:
                split_path = dataset.path / "data" / f"{split}.json"
            if not split_path.exists():
                split_path = dataset.path / f"{split}.json"
        elif dataset.type == DatasetType.DOCMATH_EVAL:
            # Handle nested structure
            split_path = dataset.path / "docmath_eval.json" / f"{split}.json"
            if not split_path.exists():
                split_path = dataset.path / f"{split}.json"
        else:
            continue

        if split_path.exists():
            return True

    return False
