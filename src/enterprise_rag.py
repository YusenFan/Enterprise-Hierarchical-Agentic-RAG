"""
EnterpriseRAG-Bench loader: streams the 1.4 GB documents parquet, keeps every document that
any question references, and reservoir-samples a fixed number of distractors per source type.
The resulting subset is cached as JSONL so runs are reproducible.
"""
import json
import logging
import os
import random
from itertools import chain
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .metadata.parsers import ENTERPRISE_SOURCE_TYPES

DOC_COLUMNS = ("doc_id", "source_type", "title", "content")
QUESTION_COLUMNS = ("question_id", "question_type", "source_types", "question",
                    "expected_doc_ids", "gold_answer", "answer_facts")
CACHE_FORMAT = "enterprise-rag-subset-v1"


def _pq():
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError("EnterpriseRAG-Bench needs pyarrow: pip install pyarrow") from e
    return pq


def documents_path(data_dir: str) -> str:
    return os.path.join(data_dir, "data", "documents", "test.parquet")


def questions_path(data_dir: str) -> str:
    return os.path.join(data_dir, "data", "questions", "test.parquet")


def _to_list(value) -> List:
    if value is None:
        return []
    return [str(v) for v in list(value)]


def read_questions(data_dir: str) -> List[Dict]:
    table = _pq().read_table(questions_path(data_dir))
    questions = []
    for row in table.to_pylist():
        questions.append({
            "question_id": row.get("question_id"),
            "question_type": row.get("question_type"),
            "source_types": _to_list(row.get("source_types")),
            "question": row.get("question") or "",
            "expected_doc_ids": _to_list(row.get("expected_doc_ids")),
            "gold_answer": row.get("gold_answer") or "",
            "answer_facts": _to_list(row.get("answer_facts")),
        })
    return questions


def iter_documents(data_dir: str, batch_size: int = 4096,
                   columns: Optional[Sequence[str]] = None) -> Iterator[Dict]:
    """Stream document rows without loading the whole content column."""
    parquet_file = _pq().ParquetFile(documents_path(data_dir))
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=list(columns or DOC_COLUMNS)):
        for row in batch.to_pylist():
            yield row


def subset_stem(subset_size: Optional[int], seed: int) -> str:
    return "enterprise_rag_full" if subset_size is None else f"enterprise_rag_n{subset_size}_s{seed}"


def subset_cache_path(cache_dir: str, subset_size: Optional[int], seed: int) -> str:
    return os.path.join(cache_dir, subset_stem(subset_size, seed) + ".jsonl")


def build_subset(data_dir: str, subset_size: Optional[int], seed: int = 42,
                 source_types: Sequence[str] = ENTERPRISE_SOURCE_TYPES,
                 batch_size: int = 4096) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Returns (documents, questions, stats). All documents in any question's `expected_doc_ids`
    are kept; the remaining budget is split evenly over `source_types` and filled by
    reservoir sampling in one streaming pass (deterministic for a fixed seed).
    `subset_size=None` keeps the whole corpus (duplicates removed).
    """
    questions = read_questions(data_dir)
    gold = set(chain.from_iterable(q["expected_doc_ids"] for q in questions))

    quota: Dict[str, int] = {}
    if subset_size is not None:
        budget = int(subset_size) - len(gold)
        if budget < 0:
            raise ValueError(
                f"subset_size={subset_size} is smaller than the {len(gold)} gold documents "
                f"referenced by the questions."
            )
        base, remainder = divmod(budget, len(source_types))
        for i, source_type in enumerate(source_types):
            quota[source_type] = base + (1 if i < remainder else 0)

    rng = random.Random(seed)
    seen = set()
    duplicates = 0
    gold_rows: List[Dict] = []
    reservoirs: Dict[str, List[Dict]] = {t: [] for t in source_types}
    seen_per_type: Dict[str, int] = {t: 0 for t in source_types}
    full: List[Dict] = []

    for row in iter_documents(data_dir, batch_size=batch_size):
        doc_id = row["doc_id"]
        if doc_id in seen:
            duplicates += 1
            continue
        seen.add(doc_id)
        if doc_id in gold:
            gold_rows.append(row)
            continue
        if subset_size is None:
            full.append(row)
            continue
        source_type = row.get("source_type")
        if source_type not in reservoirs:
            continue
        k = quota.get(source_type, 0)
        if k <= 0:
            continue
        seen_per_type[source_type] += 1
        if len(reservoirs[source_type]) < k:
            reservoirs[source_type].append(row)
        else:
            j = rng.randrange(seen_per_type[source_type])
            if j < k:
                reservoirs[source_type][j] = row

    documents = gold_rows + (full if subset_size is None else list(chain.from_iterable(reservoirs.values())))
    documents.sort(key=lambda r: (r.get("source_type") or "", r["doc_id"]))

    missing_gold = sorted(gold - {r["doc_id"] for r in gold_rows})
    if missing_gold:
        logging.warning(f"{len(missing_gold)} gold documents were not found in the corpus: {missing_gold[:5]}")

    per_type = {}
    for row in documents:
        per_type[row.get("source_type")] = per_type.get(row.get("source_type"), 0) + 1
    stats = {
        "num_documents": len(documents),
        "num_gold": len(gold_rows),
        "num_questions": len(questions),
        "duplicates_skipped": duplicates,
        "missing_gold": len(missing_gold),
        "quota": quota,
        "per_source_type": dict(sorted(per_type.items())),
    }
    return documents, questions, stats


def _write_cache(path: str, documents: List[Dict], questions: List[Dict], meta: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for q in questions:
            f.write(json.dumps({"kind": "question", **q}, ensure_ascii=False) + "\n")
        for d in documents:
            f.write(json.dumps({"kind": "document", **{k: d.get(k) for k in DOC_COLUMNS}}, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _read_cache(path: str, expected_meta: Dict) -> Optional[Tuple[List[Dict], List[Dict], Dict]]:
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        try:
            meta = json.loads(header).get("meta", {})
        except json.JSONDecodeError:
            return None
        for key, value in expected_meta.items():
            if meta.get(key) != value:
                return None
        documents, questions = [], []
        for line in f:
            row = json.loads(line)
            kind = row.pop("kind", None)
            if kind == "question":
                questions.append(row)
            elif kind == "document":
                documents.append(row)
    return documents, questions, meta.get("stats", {})


def load_enterprise_rag(data_dir: str, subset_size: Optional[int] = 5000, seed: int = 42,
                        cache_dir: Optional[str] = None, force: bool = False,
                        verbose: bool = True) -> Tuple[List[Dict], List[Dict], Dict]:
    """Load (documents, questions, stats) for a reproducible subset, building the cache on first use."""
    cache_dir = cache_dir or os.path.join(data_dir, "subsets")
    path = subset_cache_path(cache_dir, subset_size, seed)
    expected_meta = {"format": CACHE_FORMAT, "subset_size": subset_size, "seed": seed}
    if not force and os.path.exists(path):
        cached = _read_cache(path, expected_meta)
        if cached is not None:
            if verbose:
                logging.info(f'Loaded EnterpriseRAG-Bench subset from "{path}".')
            return cached
    if verbose:
        print(f"Building EnterpriseRAG-Bench subset (size={subset_size}, seed={seed}); streaming the corpus...")
    documents, questions, stats = build_subset(data_dir, subset_size, seed)
    _write_cache(path, documents, questions, {**expected_meta, "stats": stats})
    if verbose:
        print(f'Subset cached to "{path}": {stats}')
    return documents, questions, stats
