"""
Bottom-up aggregation of document metadata into abstract nodes, plus the helpers that render
metadata for the retrieval context and answer sources.

`aggregate_tree_metadata` is the single place abstract nodes receive `source_refs`,
`source_document_ids` and `aggregated_metadata`; it runs once after any tree builder finishes.
"""
from collections import Counter
from typing import Dict, List, Optional, Sequence, Set

from .document import Document, TimeRange

DEFAULT_CAPS = {
    "authors": 20, "participants": 30, "projects": 20, "entities": 30,
    "ticket_keys": 30, "channels": 10, "source_document_ids": 50,
}
DEFAULT_SOURCE_AUTHORITY = {
    "confluence": 5, "google_drive": 4, "jira": 3, "linear": 3, "github": 3,
    "gmail": 2, "fireflies": 2, "hubspot": 2, "slack": 1,
}


def _top(counter: Counter, cap: int) -> List[str]:
    return [name for name, _ in sorted(counter.items(), key=lambda x: (-x[1], x[0]))[:cap]]


def aggregate_documents(docs: Sequence[Document], num_chunks: int = 0,
                        caps: Optional[Dict[str, int]] = None,
                        source_authority: Optional[Dict[str, int]] = None) -> Dict:
    """Aggregate the metadata of `docs` (each document counted once) into one dict."""
    caps = {**DEFAULT_CAPS, **(caps or {})}
    authority = {**DEFAULT_SOURCE_AUTHORITY, **(source_authority or {})}
    counters = {k: Counter() for k in ("authors", "participants", "projects", "entities", "ticket_keys", "channels")}
    source_types: Set[str] = set()
    time_range = TimeRange()
    latest_updated = None
    for doc in docs:
        md = doc.metadata
        source_types.add(doc.source_type)
        counters["authors"].update(set(md.authors))
        counters["participants"].update(set(md.participants))
        counters["projects"].update(set(md.projects))
        counters["entities"].update(set(md.entities))
        counters["ticket_keys"].update(set(md.ticket_keys))
        if md.channel:
            counters["channels"][md.channel] += 1
        doc_range = md.time_range or TimeRange(md.created_at, md.updated_at)
        time_range = time_range.merge(doc_range)
        if md.updated_at and (latest_updated is None or md.updated_at > latest_updated):
            latest_updated = md.updated_at
    return {
        "num_documents": len(docs),
        "num_chunks": int(num_chunks),
        "source_types": sorted(source_types),
        **{key: _top(counter, caps[key]) for key, counter in counters.items()},
        "time_range": time_range.to_dict(),
        "latest_updated_at": latest_updated,
        "source_authority": max((authority.get(st, 0) for st in source_types), default=0),
    }


def node_document_ids(node) -> List[str]:
    document_id = getattr(node, "document_id", None)
    if document_id:
        return [document_id]
    return list(getattr(node, "source_document_ids", None) or [])


def aggregate_tree_metadata(tree, registry: Dict[str, Document],
                            source_authority: Optional[Dict[str, int]] = None,
                            caps: Optional[Dict[str, int]] = None) -> None:
    """
    Fill `Document.chunk_ids` for every leaf and `source_refs` / `source_document_ids` /
    `aggregated_metadata` for every node with children. Works for any builder because it only
    relies on `Node.children` (resolved recursively with memoisation, independent of layer order).
    """
    all_nodes = tree.all_nodes
    for doc in registry.values():
        doc.chunk_ids = []
    for node in all_nodes.values():
        if not node.children and getattr(node, "document_id", None) in registry:
            registry[node.document_id].chunk_ids.append(node.index)
    for doc in registry.values():
        doc.chunk_ids.sort()

    memo: Dict[int, Dict[str, Set[int]]] = {}

    def refs_for(index: int) -> Dict[str, Set[int]]:
        if index in memo:
            return memo[index]
        node = all_nodes[index]
        refs: Dict[str, Set[int]] = {}
        if not node.children:
            document_id = getattr(node, "document_id", None)
            if document_id:
                refs[document_id] = {node.index}
        else:
            for child in node.children:
                for document_id, chunk_ids in refs_for(child).items():
                    refs.setdefault(document_id, set()).update(chunk_ids)
        memo[index] = refs
        return refs

    for index in sorted(all_nodes):
        node = all_nodes[index]
        if not node.children:
            continue
        refs = refs_for(index)
        node.source_refs = [
            {"document_id": document_id, "chunk_ids": sorted(chunk_ids)}
            for document_id, chunk_ids in sorted(refs.items())
        ]
        node.source_document_ids = sorted(refs)
        node.aggregated_metadata = aggregate_documents(
            [registry[d] for d in node.source_document_ids if d in registry],
            num_chunks=sum(len(c) for c in refs.values()),
            caps=caps,
            source_authority=source_authority,
        )
    tree.documents = registry


def format_context_header(node, doc: Optional[Document] = None) -> str:
    """One-line provenance header prepended to a retrieved chunk / summary."""
    if node.children:
        agg = getattr(node, "aggregated_metadata", None) or {}
        tr = agg.get("time_range") or {}
        span = f"{tr.get('start') or 'n/a'}..{tr.get('end') or 'n/a'}"
        return f"[summary | {agg.get('num_documents', len(node_document_ids(node)))} docs | " \
               f"{', '.join(agg.get('source_types', [])) or 'n/a'} | {span}]"
    document_id = getattr(node, "document_id", None) or "n/a"
    if doc is None:
        return f"[doc: {document_id}]"
    md = doc.metadata
    date = md.created_at or (md.time_range.start if md.time_range else None) or "n/a"
    who = (md.authors[0] if md.authors else None) or (f"#{md.channel}" if md.channel else None) or "n/a"
    return f"[doc: {document_id} | {doc.source_type} | {doc.title} | {date} | {who}]"


def collect_sources(tree, node_scores: Dict[int, float], limit: Optional[int] = None) -> List[Dict]:
    """
    Map retrieved node indices (with scores) to unique source documents, best score first.
    Abstract nodes contribute all of their source documents.
    """
    registry = getattr(tree, "documents", None) or {}
    best: Dict[str, float] = {}
    order: List[str] = []
    for index, score in node_scores.items():
        node = tree.all_nodes[index]
        for document_id in node_document_ids(node):
            if document_id not in best:
                order.append(document_id)
                best[document_id] = score
            elif score is not None and score > best[document_id]:
                best[document_id] = score
    ranked = sorted(order, key=lambda d: (-(best[d] if best[d] is not None else 0.0), order.index(d)))
    sources = []
    for document_id in ranked[:limit]:
        doc = registry.get(document_id)
        sources.append({
            "document_id": document_id,
            "source_type": doc.source_type if doc else None,
            "title": doc.title if doc else None,
            "best_score": best[document_id],
        })
    return sources
