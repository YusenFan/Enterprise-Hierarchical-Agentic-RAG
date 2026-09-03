"""
Candidate generation for the hybrid score: a cached, L2-normalised dense matrix over every node of
the tree (collapsed search across all layers), the BM25 score vector over leaves (propagated to
abstract nodes as the max over their leaf descendants) and the pool assembly keyed by node index.
"""
import logging
import os
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .scoring import Candidate, TreeRelations


class DenseIndex:
    """Stacked node embeddings (rows = nodes with an embedding), normalised for cosine similarity."""

    def __init__(self, matrix: np.ndarray, row_to_node: np.ndarray, layer_of_row: np.ndarray) -> None:
        self.matrix = matrix
        self.row_to_node = row_to_node
        self.layer_of_row = layer_of_row
        self.node_to_row: Dict[int, int] = {int(n): r for r, n in enumerate(row_to_node.tolist())}

    @classmethod
    def build(cls, tree, node_to_layer: Dict[int, int], dtype: str = "float32") -> "DenseIndex":
        rows = sorted(int(i) for i, node in tree.all_nodes.items() if node.embeddings is not None)
        if not rows:
            raise ValueError("DenseIndex: no node carries an embedding.")
        matrix = np.stack([np.asarray(tree.all_nodes[i].embeddings, dtype=np.float32).ravel() for i in rows])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = (matrix / norms).astype(np.float16 if dtype == "float16" else np.float32)
        return cls(matrix, np.asarray(rows, dtype=np.int64),
                   np.asarray([node_to_layer.get(i, 0) for i in rows], dtype=np.int64))

    @classmethod
    def load_or_build(cls, tree, node_to_layer: Dict[int, int], path: Optional[str],
                      dtype: str = "float32") -> "DenseIndex":
        n_expected = sum(1 for node in tree.all_nodes.values() if node.embeddings is not None)
        if path and os.path.exists(path):
            try:
                data = np.load(path)
                if data["row_to_node"].shape[0] == n_expected:
                    return cls(data["matrix"], data["row_to_node"], data["layer_of_row"])
                logging.warning(f'Dense sidecar "{path}" does not match the tree; rebuilding.')
            except Exception:
                logging.exception(f'Dense sidecar "{path}" unreadable; rebuilding.')
        index = cls.build(tree, node_to_layer, dtype)
        if path:
            try:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                np.savez(path, matrix=index.matrix, row_to_node=index.row_to_node, layer_of_row=index.layer_of_row)
            except OSError:
                logging.exception(f'Could not write dense sidecar "{path}".')
        return index

    def sims(self, query_embedding) -> np.ndarray:
        q = np.asarray(query_embedding, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(q))
        if norm > 0:
            q = q / norm
        # Accelerate/BLAS on macOS may emit a spurious "invalid value in matmul" warning for float32
        # sgemv even with finite inputs; the values are correct, so silence it and guard NaNs.
        with np.errstate(invalid="ignore", over="ignore"):
            sims = self.matrix @ q.astype(self.matrix.dtype)
        return np.nan_to_num(np.asarray(sims, dtype=np.float32), nan=-1.0, posinf=1.0, neginf=-1.0)

    def node_sim(self, sims: np.ndarray, node_index: int) -> float:
        row = self.node_to_row.get(int(node_index))
        return float(sims[row]) if row is not None else 0.0

    def top_nodes(self, sims: np.ndarray, top_n: int, node_mask: Optional[np.ndarray] = None
                  ) -> List[Tuple[int, float]]:
        scores = sims.astype(np.float32, copy=True)
        if node_mask is not None:
            row_mask = node_mask[self.row_to_node]
            scores[~row_mask] = -np.inf
        top_n = max(1, min(int(top_n), scores.shape[0]))
        part = np.argpartition(-scores, top_n - 1)[:top_n]
        order = part[np.argsort(-scores[part], kind="stable")]
        return [(int(self.row_to_node[r]), float(scores[r])) for r in order if np.isfinite(scores[r])]


class SparseScorer:
    """BM25 score vector over the corpus (row == leaf node index) with abstract propagation."""

    def __init__(self, model, stemmer) -> None:
        self.model = model
        self.stemmer = stemmer
        self._lock = threading.Lock()

    def scores(self, query: str) -> Optional[np.ndarray]:
        if self.model is None or not hasattr(self.model, "vocab_dict"):
            return None
        import bm25s
        tokens = bm25s.tokenize(query, stemmer=self.stemmer, return_ids=False, show_progress=False)
        tokens = [t for t in (tokens[0] if tokens else []) if t in self.model.vocab_dict]
        num_docs = int(self.model.scores["num_docs"])
        if not tokens:
            return np.zeros(num_docs, dtype=np.float32)
        with self._lock:   # bm25s' numba scorer is not guaranteed thread-safe
            return np.asarray(self.model.get_scores(tokens), dtype=np.float32)


def leaf_descendants(relations: TreeRelations, tree, index: int, cache: Dict[int, List[int]]) -> List[int]:
    if index in cache:
        return cache[index]
    node = tree.all_nodes[index]
    if not node.children:
        cache[index] = [index]
        return cache[index]
    out: List[int] = []
    for child in node.children:
        out.extend(leaf_descendants(relations, tree, child, cache))
    cache[index] = out
    return out


def build_candidate_pool(tree, dense_index: DenseIndex, sims: np.ndarray, bm25_vector: Optional[np.ndarray],
                         node_to_layer: Dict[int, int], relations: TreeRelations, leaf_cache: Dict[int, List[int]],
                         dense_top_n: int, sparse_top_n: int, node_mask: Optional[np.ndarray] = None,
                         seed_nodes: Optional[Sequence[int]] = None) -> Dict[int, Candidate]:
    """
    Pool keyed by node index. Dense seeds = `seed_nodes` (traversal mode) or the masked top-N of
    `sims` (collapsed mode); sparse seeds = top-N leaves of `bm25_vector` plus their parents.
    Every member gets its exact dense similarity and BM25 score (max over leaf descendants).
    """
    pool: Dict[int, Candidate] = {}

    def allowed(index: int) -> bool:
        return node_mask is None or (index < node_mask.shape[0] and bool(node_mask[index]))

    if seed_nodes is None:
        for index, sim in dense_index.top_nodes(sims, dense_top_n, node_mask):
            pool[index] = Candidate(index, node_to_layer.get(index, 0), dense=sim, origin="dense")
    else:
        for index in seed_nodes:
            if allowed(index) and index not in pool:
                pool[index] = Candidate(index, node_to_layer.get(index, 0),
                                        dense=dense_index.node_sim(sims, index), origin="traversal")

    if bm25_vector is not None and bm25_vector.size:
        vec = bm25_vector
        if node_mask is not None:
            vec = vec.copy()
            n = min(vec.shape[0], node_mask.shape[0])
            vec[:n][~node_mask[:n]] = 0.0
        top_n = max(1, min(int(sparse_top_n), vec.shape[0]))
        part = np.argpartition(-vec, top_n - 1)[:top_n]
        for leaf in part[np.argsort(-vec[part], kind="stable")]:
            leaf = int(leaf)
            if vec[leaf] <= 0 or leaf not in tree.all_nodes:
                continue
            if leaf not in pool:
                pool[leaf] = Candidate(leaf, node_to_layer.get(leaf, 0), dense=dense_index.node_sim(sims, leaf),
                                       origin="sparse")
            else:
                pool[leaf].origin = "both"
            parent = relations.parent.get(leaf)
            if parent is not None and allowed(parent) and parent not in pool:
                pool[parent] = Candidate(parent, node_to_layer.get(parent, 0),
                                         dense=dense_index.node_sim(sims, parent), origin="sparse-parent")
        for cand in pool.values():
            if cand.layer == 0:
                cand.sparse = float(bm25_vector[cand.node_index]) if cand.node_index < bm25_vector.shape[0] else 0.0
            else:
                leaves = [l for l in leaf_descendants(relations, tree, cand.node_index, leaf_cache)
                          if l < bm25_vector.shape[0]]
                cand.sparse = float(bm25_vector[leaves].max()) if leaves else 0.0
    return pool
