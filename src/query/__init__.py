"""
Query-side metadata use for the hierarchical tree (phase 2).

- understanding.py: structured constraints extracted from a question (rules and / or one LLM call).
- time_expressions.py: query-only date windows ("Q4 2025", "early March 2026", ...).
- scoring.py: metadata / level / redundancy terms of the hybrid score and MMR selection.
- candidates.py: cached dense matrix, BM25 score vector, hard-filter masks.
"""
from .understanding import QueryConstraints, QueryUnderstanding, parse_query_rules, vocabulary_for_tree
from .time_expressions import query_time_window

__all__ = [
    "QueryConstraints", "QueryUnderstanding", "parse_query_rules", "vocabulary_for_tree",
    "query_time_window",
]
