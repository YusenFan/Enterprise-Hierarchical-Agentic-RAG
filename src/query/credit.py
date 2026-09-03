"""
Document credit for retrieval evaluation: which documents does a ranked list of nodes "retrieve"?
Leaves credit their document; abstract nodes credit their source documents only when they cover
at most `max_abstract_docs` documents (a root spanning 30 documents must not earn recall by
being broad). A leaf-only variant is reported next to it.
"""
from typing import Dict, List, Optional, Sequence, Tuple

from ..metadata.aggregate import collect_sources


def merge_node_scores(layer_infos: Sequence[Sequence[Dict]], top_k: int, all_layers: bool = False) -> Dict[int, float]:
    """
    Best score per node over one or several retrievals (agentic sub-questions), sorted, top_k.
    `all_layers=False` keeps the legacy behaviour (leaf nodes only).
    """
    scores: Dict[int, float] = {}
    for layer_information in layer_infos or []:
        for node in layer_information or []:
            if not all_layers and node.get("layer_number", 0) != 0:
                continue
            index, score = node["node_index"], node.get("score")
            score = float(score) if score is not None else 0.0
            if index not in scores or score > scores[index]:
                scores[index] = score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return dict(ranked[:top_k] if top_k is not None else ranked)


def document_credit(tree, node_scores: Dict[int, float], max_abstract_docs: Optional[int] = None
                    ) -> Tuple[List[Dict], List[Dict]]:
    """(sources with capped abstract expansion, leaf-only sources), both best score first."""
    sources = collect_sources(tree, node_scores, max_abstract_docs=max_abstract_docs)
    leaf_scores = {index: score for index, score in node_scores.items()
                   if index in tree.all_nodes and not tree.all_nodes[index].children}
    leaf_only = collect_sources(tree, leaf_scores)
    return sources, leaf_only
