"""
Score terms of the metadata-aware hybrid score and the greedy (MMR) selection:

    Score(q, n) = alpha*S_dense + beta*S_BM25 + gamma*S_metadata + delta*S_level - lambda*S_redundancy

- `MetadataIndex` turns every node into a flat `NodeMetadataView` (leaf: its Document +
  local_metadata; abstract: aggregated_metadata) and keeps the arrays needed by the hard filter.
- `metadata_match` scores a view against `QueryConstraints`, field by field, over the fields the
  query actually specifies (a query without constraints scores 0 everywhere).
- `level_score` maps the benchmark question_type to a layer preference.
- `TreeRelations` answers "is a an ancestor / descendant of b" for the redundancy penalty.
- `select_with_mmr` normalises the dense / BM25 terms over the pool and picks top_k greedily.
"""
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..metadata.parsers import ENTERPRISE_SOURCE_TYPES
from .understanding import QueryConstraints

SOURCE_BIT = {src: 1 << i for i, src in enumerate(ENTERPRISE_SOURCE_TYPES)}
DEFAULT_FIELD_WEIGHTS = {
    "ticket_keys": 3.0, "time": 2.0, "source_type": 2.0, "projects": 2.0,
    "entities": 1.5, "people": 1.5, "channels": 1.0,
}
DEFAULT_LEVEL_PREFERENCE = {
    "high_level": {"0": 0.2, "1": 0.6, "2+": 1.0},
    "completeness": {"0": 0.2, "1": 0.6, "2+": 1.0},
    "project_related": {"0": 0.8, "1": 0.8, "2+": 0.4},
    "miscellaneous": {"0": 0.8, "1": 0.8, "2+": 0.4},
    "default": {"0": 1.0, "1": 0.5, "2+": 0.0},
}
DEFAULT_WEIGHTS = {"alpha": 1.0, "beta": 0.5, "gamma": 0.5, "delta": 0.3, "lambda": 0.3}


def _norm(value) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _ordinal(iso: Optional[str]) -> Optional[int]:
    if not iso or not isinstance(iso, str) or len(iso) < 10:
        return None
    try:
        return date.fromisoformat(iso[:10]).toordinal()
    except ValueError:
        return None


# ------------------------------------------------------------------------- node views ----------

@dataclass
class NodeMetadataView:
    node_index: int
    layer: int
    document_ids: List[str] = field(default_factory=list)
    num_documents: int = 0
    source_types: Set[str] = field(default_factory=set)
    people: Set[str] = field(default_factory=set)          # normalised full names
    projects: Set[str] = field(default_factory=set)        # normalised
    entities: Set[str] = field(default_factory=set)        # normalised
    ticket_keys: Set[str] = field(default_factory=set)     # upper-case keys
    channels: Set[str] = field(default_factory=set)
    t_start: Optional[int] = None                          # ordinal days, inclusive
    t_end: Optional[int] = None

    @property
    def is_leaf(self) -> bool:
        return self.layer == 0


def _doc_interval(md) -> Tuple[Optional[int], Optional[int]]:
    start = end = None
    tr = getattr(md, "time_range", None)
    if tr is not None:
        start, end = _ordinal(getattr(tr, "start", None)), _ordinal(getattr(tr, "end", None))
    if start is None:
        start = _ordinal(getattr(md, "created_at", None))
    if end is None:
        end = _ordinal(getattr(md, "updated_at", None))
    if start is None and end is None:
        return None, None
    start = start if start is not None else end
    end = end if end is not None else start
    return min(start, end), max(start, end)


def document_view(doc) -> Dict:
    """Flat, normalised metadata of one Document (computed once per tree, shared by its leaves)."""
    md = doc.metadata
    people = {_norm(p) for p in list(md.authors or []) + list(md.participants or []) if _norm(p)}
    t_start, t_end = _doc_interval(md)
    channels = set()
    if md.channel:
        channels.add(_norm(md.channel))
    for c in (md.extra or {}).get("slack_channels", []) or []:
        channels.add(_norm(str(c).lstrip("#")))
    return {
        "source_types": {doc.source_type},
        "people": people,
        "projects": {_norm(p) for p in md.projects or [] if _norm(p)},
        "entities": {_norm(e) for e in md.entities or [] if _norm(e)} | {_norm(doc.title)} - {""},
        "ticket_keys": {str(k).upper() for k in md.ticket_keys or []},
        "channels": channels - {""},
        "t_start": t_start,
        "t_end": t_end,
    }


def _leaf_people(local_metadata: Optional[Dict]) -> Set[str]:
    out = set()
    if not local_metadata:
        return out
    for key in ("speakers", "from"):
        value = local_metadata.get(key)
        if isinstance(value, str):
            value = [value]
        for v in value or []:
            n = _norm(v)
            if n:
                out.add(n)
    return out


class MetadataIndex:
    """Per-tree metadata views for every node + arrays for corpus-wide hard filtering."""

    def __init__(self, tree, node_to_layer: Dict[int, int]) -> None:
        self.tree = tree
        self.node_to_layer = node_to_layer
        self.documents = getattr(tree, "documents", None) or {}
        self._doc_views: Dict[str, Dict] = {}
        self.views: Dict[int, NodeMetadataView] = {}
        self.size = (max(tree.all_nodes.keys()) + 1) if tree.all_nodes else 0
        self.src_bits = np.zeros(self.size, dtype=np.int64)
        self.t_start = np.full(self.size, -1, dtype=np.int64)
        self.t_end = np.full(self.size, -1, dtype=np.int64)
        self.ticket_nodes: Dict[str, Set[int]] = {}
        for index, node in tree.all_nodes.items():
            view = self._build_view(index, node)
            self.views[index] = view
            bits = 0
            for src in view.source_types:
                bits |= SOURCE_BIT.get(src, 0)
            self.src_bits[index] = bits
            if view.t_start is not None:
                self.t_start[index] = view.t_start
                self.t_end[index] = view.t_end
            for key in view.ticket_keys:
                self.ticket_nodes.setdefault(key, set()).add(index)

    def _doc_view(self, document_id: str) -> Optional[Dict]:
        if document_id in self._doc_views:
            return self._doc_views[document_id]
        doc = self.documents.get(document_id)
        view = document_view(doc) if doc is not None else None
        self._doc_views[document_id] = view
        return view

    def _build_view(self, index: int, node) -> NodeMetadataView:
        layer = self.node_to_layer.get(index, 0)
        if not node.children:
            document_id = getattr(node, "document_id", None)
            view = NodeMetadataView(index, layer, document_ids=[document_id] if document_id else [],
                                    num_documents=1 if document_id else 0)
            dv = self._doc_view(document_id) if document_id else None
            if dv:
                view.source_types = set(dv["source_types"])
                view.people = set(dv["people"]) | _leaf_people(getattr(node, "local_metadata", None))
                view.projects, view.entities = set(dv["projects"]), set(dv["entities"])
                view.ticket_keys, view.channels = set(dv["ticket_keys"]), set(dv["channels"])
                view.t_start, view.t_end = dv["t_start"], dv["t_end"]
            else:
                view.people = _leaf_people(getattr(node, "local_metadata", None))
            return view
        agg = getattr(node, "aggregated_metadata", None) or {}
        doc_ids = list(getattr(node, "source_document_ids", None) or [])
        view = NodeMetadataView(index, layer, document_ids=doc_ids,
                                num_documents=int(agg.get("num_documents") or len(doc_ids)))
        view.source_types = set(agg.get("source_types") or [])
        view.people = {_norm(p) for p in list(agg.get("authors") or []) + list(agg.get("participants") or []) if _norm(p)}
        view.projects = {_norm(p) for p in agg.get("projects") or [] if _norm(p)}
        view.entities = {_norm(e) for e in agg.get("entities") or [] if _norm(e)}
        view.ticket_keys = {str(k).upper() for k in agg.get("ticket_keys") or []}
        view.channels = {_norm(c) for c in agg.get("channels") or [] if _norm(c)}
        tr = agg.get("time_range") or {}
        s, e = _ordinal(tr.get("start")), _ordinal(tr.get("end"))
        if s is not None or e is not None:
            s = s if s is not None else e
            e = e if e is not None else s
            view.t_start, view.t_end = min(s, e), max(s, e)
        return view

    def view(self, index: int) -> NodeMetadataView:
        return self.views[index]

    # ---- hard filter ---------------------------------------------------------------------------
    def hard_mask(self, constraints: QueryConstraints, fields: Sequence[str], tol_days: int) -> Tuple[np.ndarray, List[str]]:
        """
        Boolean mask over node indices: False = contradicts a specified hard constraint.
        Nodes with unknown metadata for a field never fail it. Returns (mask, fields_applied).
        """
        mask = np.ones(self.size, dtype=bool)
        applied: List[str] = []
        if constraints is None:
            return mask, applied
        if "source_type" in fields and constraints.source_types:
            wanted = 0
            for src in constraints.source_types:
                wanted |= SOURCE_BIT.get(src, 0)
            if wanted:
                known = self.src_bits != 0
                mask &= ~known | ((self.src_bits & wanted) != 0)
                applied.append("source_type")
        if "time" in fields and constraints.time_range:
            qs, qe = _ordinal(constraints.time_range.get("start")), _ordinal(constraints.time_range.get("end"))
            if qs is not None and qe is not None:
                qs, qe = min(qs, qe) - int(tol_days), max(qs, qe) + int(tol_days)
                known = self.t_start >= 0
                overlap = (self.t_end >= qs) & (self.t_start <= qe)
                mask &= ~known | overlap
                applied.append("time")
        if "ticket_keys" in fields and constraints.ticket_keys:
            wanted_nodes: Set[int] = set()
            for key in constraints.ticket_keys:
                wanted_nodes |= self.ticket_nodes.get(str(key).upper(), set())
            if wanted_nodes:
                # only nodes that mention *some* ticket key are "known" for this field
                known = np.zeros(self.size, dtype=bool)
                for nodes in self.ticket_nodes.values():
                    known[list(nodes)] = True
                ok = np.zeros(self.size, dtype=bool)
                ok[list(wanted_nodes)] = True
                mask &= ~known | ok
                applied.append("ticket_keys")
        return mask, applied


# --------------------------------------------------------------------------- matching ----------

def _person_match(query_name: str, people: Set[str]) -> bool:
    q = _norm(query_name)
    if not q or not people:
        return False
    if q in people:
        return True
    q_tokens = q.split()
    for name in people:
        tokens = name.split()
        if not tokens:
            continue
        if len(q_tokens) == 1:
            if q_tokens[0] in (tokens[0], tokens[-1]):
                return True
        elif q_tokens[-1] == tokens[-1] and q_tokens[0][0] == tokens[0][0]:
            return True
    return False


def _fraction(values: Iterable[str], pool: Set[str], substring: bool = False) -> Optional[float]:
    values = [v for v in (_norm(x) for x in values) if v]
    if not values:
        return None
    hits = 0
    for v in values:
        if v in pool or (substring and any(v in p or p in v for p in pool)):
            hits += 1
    return hits / len(values)


def time_overlap(q_start: int, q_end: int, n_start: Optional[int], n_end: Optional[int], tol_days: int = 0) -> float:
    """Overlap coefficient |Q ∩ N| / min(|Q|, |N|) on inclusive day intervals; 0 when unknown."""
    if n_start is None or n_end is None:
        return 0.0
    qs, qe = min(q_start, q_end) - int(tol_days), max(q_start, q_end) + int(tol_days)
    inter = min(qe, n_end) - max(qs, n_start) + 1
    if inter <= 0:
        return 0.0
    return float(min(1.0, inter / min(qe - qs + 1, n_end - n_start + 1)))


def metadata_match(constraints: QueryConstraints, view: NodeMetadataView,
                   field_weights: Optional[Dict[str, float]] = None, tol_days: int = 7
                   ) -> Tuple[float, Dict[str, float]]:
    """Weighted mean of per-field matches over the fields the query specifies (0 when none)."""
    if constraints is None:
        return 0.0, {}
    weights = dict(DEFAULT_FIELD_WEIGHTS)
    weights.update(field_weights or {})
    per_field: Dict[str, float] = {}
    if constraints.source_types:
        per_field["source_type"] = 1.0 if view.source_types & set(constraints.source_types) else 0.0
    if constraints.time_range:
        qs, qe = _ordinal(constraints.time_range.get("start")), _ordinal(constraints.time_range.get("end"))
        if qs is not None and qe is not None:
            per_field["time"] = time_overlap(qs, qe, view.t_start, view.t_end, tol_days)
    if constraints.people:
        matched = sum(1 for p in constraints.people if _person_match(p, view.people))
        per_field["people"] = matched / len(constraints.people)
    for name, substring in (("projects", True), ("entities", True), ("channels", False)):
        values = getattr(constraints, name)
        if values:
            frac = _fraction(values, getattr(view, name), substring=substring)
            if frac is not None:
                per_field[name] = frac
    if constraints.ticket_keys:
        wanted = {str(k).upper() for k in constraints.ticket_keys}
        per_field["ticket_keys"] = len(wanted & view.ticket_keys) / len(wanted)
    if not per_field:
        return 0.0, per_field
    total = sum(weights.get(f, 1.0) for f in per_field)
    score = sum(weights.get(f, 1.0) * v for f, v in per_field.items()) / total if total > 0 else 0.0
    return float(score), per_field


def level_score(question_type: Optional[str], layer: int,
                level_preference: Optional[Dict[str, Dict[str, float]]] = None) -> float:
    prefs = level_preference or DEFAULT_LEVEL_PREFERENCE
    table = prefs.get(question_type or "") or prefs.get("default")
    if not table:
        return 0.5
    key = "0" if layer <= 0 else ("1" if layer == 1 else "2+")
    return float(table.get(key, table.get(str(layer), 0.5)))


# --------------------------------------------------------------------------- redundancy --------

class TreeRelations:
    """Parent map + memoised ancestor sets; `related(a, b)` in {0, 0.5, 1}."""

    def __init__(self, tree, views: Optional[Dict[int, NodeMetadataView]] = None) -> None:
        self.parent: Dict[int, int] = {}
        for index, node in tree.all_nodes.items():
            for child in node.children or ():
                self.parent[child] = index
        self._ancestors: Dict[int, Set[int]] = {}
        self.views = views

    def ancestors(self, index: int) -> Set[int]:
        if index in self._ancestors:
            return self._ancestors[index]
        out: Set[int] = set()
        cur = index
        while cur in self.parent:
            cur = self.parent[cur]
            if cur in out:
                break
            out.add(cur)
        self._ancestors[index] = out
        return out

    def related(self, a: int, b: int) -> float:
        if a == b or b in self.ancestors(a) or a in self.ancestors(b):
            return 1.0
        if self.views is None:
            return 0.0
        va, vb = self.views.get(a), self.views.get(b)
        if va is None or vb is None or not va.document_ids or not vb.document_ids:
            return 0.0
        sa, sb = set(va.document_ids), set(vb.document_ids)
        if va.is_leaf and vb.is_leaf:
            return 0.5 if sa == sb else 0.0
        inter = len(sa & sb)
        if inter == 0:
            return 0.0
        return 0.5 if inter / len(sa | sb) > 0.5 else 0.0


# ------------------------------------------------------------------------------ selection ------

@dataclass
class Candidate:
    node_index: int
    layer: int
    dense: float = 0.0          # raw cosine similarity
    sparse: float = 0.0         # raw BM25 score
    metadata: float = 0.0
    level: float = 0.0
    meta_fields: Dict[str, float] = field(default_factory=dict)
    origin: str = ""


def normalize_scores(values: Sequence[float], method: str = "minmax") -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    if method == "rank":
        order = np.argsort(-arr, kind="stable")
        ranks = np.empty(arr.size, dtype=np.float64)
        ranks[order] = np.arange(arr.size)
        return 1.0 - ranks / max(arr.size - 1, 1)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.full(arr.size, 0.5 if arr.size > 1 else 1.0)
    return (arr - lo) / (hi - lo)


def select_with_mmr(candidates: List[Candidate], weights: Optional[Dict[str, float]], top_k: int,
                    relations: Optional[TreeRelations] = None, score_norm: str = "minmax"
                    ) -> List[Tuple[Candidate, float, Dict[str, float]]]:
    """
    Greedy selection: at each step pick argmax(base - lambda * max_{j selected} related(i, j)).
    Returns [(candidate, final_score, sub_scores)] in selection order.
    """
    if not candidates:
        return []
    w = dict(DEFAULT_WEIGHTS)
    w.update(weights or {})
    dense = normalize_scores([c.dense for c in candidates], score_norm)
    sparse_raw = [c.sparse for c in candidates]
    sparse = normalize_scores(sparse_raw, score_norm) if any(s > 0 for s in sparse_raw) else np.zeros(len(candidates))
    base = (w["alpha"] * dense + w["beta"] * sparse
            + w["gamma"] * np.asarray([c.metadata for c in candidates])
            + w["delta"] * np.asarray([c.level for c in candidates]))
    lam = float(w.get("lambda", 0.0))
    remaining = list(range(len(candidates)))
    selected: List[int] = []
    out: List[Tuple[Candidate, float, Dict[str, float]]] = []
    while remaining and len(out) < top_k:
        best_i, best_score, best_pen = None, -np.inf, 0.0
        for i in remaining:
            pen = 0.0
            if lam > 0 and relations is not None and selected:
                pen = max(relations.related(candidates[i].node_index, candidates[j].node_index) for j in selected)
            score = float(base[i] - lam * pen)
            if score > best_score:
                best_i, best_score, best_pen = i, score, pen
        remaining.remove(best_i)
        selected.append(best_i)
        c = candidates[best_i]
        out.append((c, best_score, {
            "dense": float(dense[best_i]), "dense_raw": float(c.dense),
            "sparse": float(sparse[best_i]), "sparse_raw": float(c.sparse),
            "metadata": float(c.metadata), "level": float(c.level), "redundancy": float(best_pen),
            "meta_fields": dict(c.meta_fields),
        }))
    return out
