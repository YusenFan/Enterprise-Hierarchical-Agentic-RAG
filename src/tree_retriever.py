import logging
import os
import shutil
from typing import Dict, Tuple, List, Set, Optional
from threading import Lock

import time
import tiktoken
import bm25s
import Stemmer
import numpy as np
from .metadata.aggregate import format_context_header
from .utils import (Node, Tree, distances_from_embeddings, get_embeddings, get_text_list,
                    get_sparse_save_name, get_tree_save_name, repair_node_indices, reverse_mapping, rrf)

logging.basicConfig(format="%(asctime)s - %(message)s", 
                    level=logging.INFO,
                    filename="./log/stdout.log",
                    filemode="a"
                    )

RETRIEVE_MODES = ("legacy", "hybrid_score")
CANDIDATE_MODES = ("collapsed", "traversal")


class TreeRetriever:

    def __init__(self, conf, tree) -> None:
        if not isinstance(tree, Tree):
            raise ValueError("tree must be an instance of Tree")

        self.conf = conf
        repair_node_indices(tree)
        if self.conf["tokenizer"] is not None and isinstance(self.conf["tokenizer"], str):
            self.conf["tokenizer"] = tiktoken.get_encoding(self.conf["tokenizer"])
        self.tree = tree
        
        if self.conf["start_layer"] is None:
            if self.conf["abstract_layer_as_context"] > 0: 
                # if context needs higher abstracts, start searching from root
                self.conf["start_layer"] = min(self.conf["abstract_layer_as_context"], self.tree.num_layers)
            else: 
                # otherwise, start searching from the first layer with more than top_k nodes
                for layer, node_idx_list in reversed(self.tree.layer_to_node_indices.items()):
                    if len(node_idx_list) >= self.conf["tree_top_k"]:
                        self.conf["start_layer"] = layer
                        break
                if self.conf["start_layer"] is None:
                    raise ValueError(
                        f"top k value ({self.conf['tree_top_k']}) is larger than "
                        f"the number of leaf nodes ({len(self.tree.layer_to_node_indices[0])})"
                    )
        elif self.conf["start_layer"] > self.tree.num_layers:
            self.conf["start_layer"] = self.tree.num_layers

        self.tree_node_index_to_layer = reverse_mapping(self.tree.layer_to_node_indices)
        
        self.stemmer = Stemmer.Stemmer("english")
        self.hybrid_search_model = bm25s.BM25() if self.conf["hybrid_search"] else None
        if self.hybrid_search_model is not None and self.conf["save_dir"] is not None:
            hybrid_save_dir = os.path.join(
                self.conf["save_dir"],
                get_sparse_save_name(self.conf),
            )
            if os.path.exists(hybrid_save_dir) and not self.conf["force_sparse_index_from_scratch"]:
                self.hybrid_search_model = self.hybrid_search_model.load(hybrid_save_dir, load_corpus=True)
                logging.info(f"Loaded vocab from \"{hybrid_save_dir}\".")

        # ---- metadata-aware hybrid score (phase 2); everything below is lazy and only used
        #      when retrieve_mode == "hybrid_score", so legacy datasets / configs are untouched.
        self.retrieve_mode = str(self.conf.get("retrieve_mode") or "legacy")
        if self.retrieve_mode not in RETRIEVE_MODES:
            raise ValueError(f'retrieve_mode must be one of {RETRIEVE_MODES}, got "{self.retrieve_mode}".')
        self.candidate_mode = str(self.conf.get("candidate_mode") or "collapsed")
        if self.candidate_mode not in CANDIDATE_MODES:
            raise ValueError(f'candidate_mode must be one of {CANDIDATE_MODES}, got "{self.candidate_mode}".')
        self._index_lock = Lock()
        self._dense_index = None
        self._meta_index = None
        self._relations = None
        self._leaf_cache: Dict[int, List[int]] = {}
        self._sparse_scorer = None
        self.query_understanding = None
        if self.retrieve_mode == "hybrid_score":
            from .query.understanding import QueryUnderstanding, vocabulary_for_tree
            if getattr(self.tree, "documents", None):
                self.query_understanding = QueryUnderstanding(self.conf, vocabulary_for_tree(self.conf, self.tree))
            else:
                # no metadata on this tree: constraints are always empty, no LLM call is made
                self.query_understanding = QueryUnderstanding({**self.conf, "query_understanding": "none"})
    
    def embed(self, text: str) -> List[float]:
        return self.conf["embed_model"].embed(text)

    # ------------------------------------------------------------------ lazy shared structures ----
    def _dense_sidecar_path(self) -> Optional[str]:
        save_dir = self.conf.get("save_dir")
        if not save_dir:
            return None
        try:
            return os.path.join(save_dir, get_tree_save_name(self.conf) + ".dense.npz")
        except KeyError:
            return None

    def _ensure_hybrid_structures(self) -> None:
        if self._dense_index is not None:
            return
        with self._index_lock:
            if self._dense_index is not None:
                return
            from .query.candidates import DenseIndex, SparseScorer
            from .query.scoring import MetadataIndex, TreeRelations
            t0 = time.time()
            meta_index = MetadataIndex(self.tree, self.tree_node_index_to_layer)
            relations = TreeRelations(self.tree, meta_index.views)
            dense_index = DenseIndex.load_or_build(
                self.tree, self.tree_node_index_to_layer, self._dense_sidecar_path(),
                dtype=str(self.conf.get("dense_index_dtype") or "float32"),
            )
            self._meta_index, self._relations = meta_index, relations
            self._dense_index = dense_index
            logging.info(f"Hybrid-score structures ready in {time.time() - t0:.1f}s "
                         f"({dense_index.matrix.shape[0]} nodes x {dense_index.matrix.shape[1]} dims).")

    @property
    def sparse_scorer(self):
        """BM25 score-vector helper (None without a hybrid index). Independent of the shared structures."""
        if self._sparse_scorer is None and self.conf["hybrid_search"] and self.hybrid_search_model is not None \
                and hasattr(self.hybrid_search_model, "vocab_dict"):
            from .query.candidates import SparseScorer
            self._sparse_scorer = SparseScorer(self.hybrid_search_model, self.stemmer)
        return self._sparse_scorer

    @property
    def dense_index(self):
        self._ensure_hybrid_structures()
        return self._dense_index

    @property
    def metadata_index(self):
        self._ensure_hybrid_structures()
        return self._meta_index

    @property
    def relations(self):
        self._ensure_hybrid_structures()
        return self._relations

    # ---------------------------------------------------------------------------- legacy ----------
    def _tree_retrieve(
        self,
        current_nodes: List[Node],
        query: str,
        start_layer: int,
        query_embedding=None,
    ) -> Tuple[List[Node], List[str], List[float]]:
        selected_nodes, context, scores, _ = self._traverse(current_nodes, query, start_layer, query_embedding)
        return selected_nodes, context, scores

    def _traverse(
        self,
        current_nodes: List[Node],
        query: str,
        start_layer: int,
        query_embedding=None,
    ) -> Tuple[List[Node], List[str], List[float], List[Tuple[int, float]]]:
        """
        Top-down traversal (the original `_tree_retrieve`), additionally returning every node that
        was ranked into `best_indices` on any layer as (node_index, distance) for the hybrid score.
        """
        if query_embedding is None:
            query_embedding = self.embed(query)
        
        selected_nodes = []
        node_list = current_nodes
        visited: List[Tuple[int, float]] = []
        scores = np.zeros(0)

        for layer in range(start_layer, -1, -1):
            
            # 1) Calculate embeddings between query and node
            embeddings = get_embeddings(node_list)
            distances = distances_from_embeddings(query_embedding, embeddings, self.conf["distance"])
            indices = np.argsort(distances)

            # 2) Remove duplicate nodes
            embeddings = np.asarray(embeddings)
            mask = np.any(embeddings[indices[1:]] != embeddings[indices[:-1]], axis=-1)
            indices = indices[np.concatenate(([True], mask))]

            if self.conf["selection_mode"] == "threshold":
                best_indices = [
                    index for index in indices if distances[index] > self.conf["threshold"]
                ]
            elif self.conf["selection_mode"] == "top_k":
                # 3) Extract top-k nodes
                best_indices = indices[: self.conf["tree_top_k"]]

            nodes_to_add = [node_list[idx] for idx in best_indices]
            visited.extend((node_list[idx].index, float(distances[idx])) for idx in best_indices)

            if layer <= self.conf["abstract_layer_as_context"]:
                selected_nodes.extend(nodes_to_add)
                
                if layer == 0:
                    if self.conf["distance"] == "cosine":
                        # normalized distance as document scores
                        scores = (2 - np.asarray(distances)) / 2
                    else:
                        raise NotImplementedError
                    scores = scores[best_indices]
            
            # 4) Add all children to the candidate set
            if layer > 0:
                child_nodes = []
                for index in best_indices:
                    child_nodes.extend(node_list[index].children)
                child_nodes = list(dict.fromkeys(child_nodes))
                node_list = [self.tree.all_nodes[i] for i in child_nodes]

        context = get_text_list(selected_nodes)
        return selected_nodes, context, scores.tolist(), visited

    def retrieve(
        self,
        query: str,
        max_tokens: int = 3500,
        tokenizer_lock: Lock = None,
        query_embedding=None,
        extras: Optional[Dict] = None,
        question_type: Optional[str] = None,
    ) -> Tuple[List[str], List[Dict], float, Dict[str, float]]:
        
        if not isinstance(query, str):
            raise ValueError("query must be a string")

        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be an integer and at least 1")

        if self.retrieve_mode == "hybrid_score":
            return self._retrieve_hybrid_score(query, query_embedding, extras, question_type)

        start_time = time.time()

        layer_nodes = [self.tree.all_nodes[idx] for idx in self.tree.layer_to_node_indices[self.conf["start_layer"]]]
        retrieved_nodes, _, scores = self._tree_retrieve(
            layer_nodes, query, self.conf["start_layer"], query_embedding=query_embedding
        )
        retrieved_node_indices = [node.index for node in retrieved_nodes]
        single_retrieval_time = time.time() - start_time

        sparse_start_time = time.time()
        if self.conf["hybrid_search"] and self.hybrid_search_model is not None:
            hybrid_node_indices = self._hybrid_retrieve(query, self.conf["sparse_top_k"])
        else:
            hybrid_node_indices = []
        sparse_time = time.time() - sparse_start_time
        
        rerank_start_time = time.time()
        if self.conf["rerank"] and self.conf["rerank_model"] is not None:
            # Reranking
            all_retrieved_docs = {self.tree.all_nodes[idx].text: self.tree.all_nodes[idx].index  
                                  for idx in retrieved_node_indices + hybrid_node_indices}
            if self.conf["rerank_batch_size"] >= self.conf["rerank_top_k"]:
                self.conf["rerank_batch_size"] = -1
            if self.conf["multithreading_qa_batch_size"] > 1:
                with tokenizer_lock:
                    rerank_scores = self.conf["rerank_model"].rerank(query=query, 
                                                                     documents=list(all_retrieved_docs.keys()),
                                                                     batch_size=self.conf["rerank_batch_size"])
            else:
                rerank_scores = self.conf["rerank_model"].rerank(query=query, 
                                                                 documents=list(all_retrieved_docs.keys()),
                                                                 batch_size=self.conf["rerank_batch_size"])
            final_node_indices = sorted(zip(all_retrieved_docs.values(), rerank_scores), 
                                        key=lambda x: x[1], 
                                        reverse=True)[:self.conf["rerank_top_k"]]
            final_node_indices, scores = zip(*final_node_indices)
            final_nodes = [self.tree.all_nodes[idx] for idx in final_node_indices]
            context = get_text_list(final_nodes)
                
        else:
            # Combine to result sets with RRF
            retrieved_docs = {self.tree.all_nodes[idx].text: self.tree.all_nodes[idx].index  
                              for idx in retrieved_node_indices}
            hybrid_docs = {self.tree.all_nodes[idx].text: self.tree.all_nodes[idx].index  
                           for idx in hybrid_node_indices}
            context, scores = rrf([list(retrieved_docs.keys()), list(hybrid_docs.keys())], 
                                  top_k=self.conf["rerank_top_k"], k=int(self.conf.get("rrf_k") or 60))
            retrieved_docs.update(hybrid_docs)
            final_node_indices = [retrieved_docs[passage] for passage in context]
            final_nodes = [self.tree.all_nodes[idx] for idx in final_node_indices]
        rerank_time = time.time() - rerank_start_time

        context = self._decorate_context(context, final_nodes)
        end_time = time.time()
        layer_information = self._layer_information(final_nodes, scores)

        return (context, layer_information, end_time - start_time, 
               {'tree': single_retrieval_time, 'sparse': sparse_time, 'rerank': rerank_time})

    # ------------------------------------------------------------------------ shared output ------
    def _decorate_context(self, context: List[str], final_nodes: List[Node]) -> List[str]:
        def _add_info(context, final_nodes):
            '''Prepend extra info like chunk ID to mark relative positions. Used for summarization.'''
            for i in range(len(context)):
                node = final_nodes[i]
                ctx = context[i]
                if len(node.children):
                    context[i] = f"[LAYER: {self.tree_node_index_to_layer[node.index]}] " + ctx
                else:
                    context[i] = f"[ID: {node.index}] " + ctx
            return context
        
        if self.conf["abstract_layer_as_context"] or self.conf["answer_type"] == "long":
            context = _add_info(context, final_nodes)

        documents = getattr(self.tree, "documents", None) or {}

        def _add_metadata_header(context, final_nodes):
            '''Prepend "[doc: id | source | title | date | author]" provenance to every chunk.'''
            for i in range(len(context)):
                node = final_nodes[i]
                doc = documents.get(getattr(node, "document_id", None))
                context[i] = format_context_header(node, doc) + "\n" + context[i]
            return context

        # metadata-in-text indexes already carry the header inside the node text (arm B)
        if self.conf.get("context_metadata_header") and not self.conf.get("enterprise_chunk_metadata_prefix"):
            context = _add_metadata_header(context, final_nodes)
        return context

    def _layer_information(self, final_nodes: List[Node], scores, sub_scores: Optional[List[Dict]] = None) -> List[Dict]:
        documents = getattr(self.tree, "documents", None) or {}
        layer_information = []
        for i, node in enumerate(final_nodes):
            document_id = getattr(node, "document_id", None)
            doc = documents.get(document_id) if document_id else None
            source_document_ids = getattr(node, "source_document_ids", None)
            entry = {
                "node_index": node.index,
                "document_index": node.document_index,
                "chunk_index": node.chunk_index,
                "layer_number": self.tree_node_index_to_layer[node.index],
                "score": scores[i],
                "document_id": document_id,
                "source_document_ids": list(source_document_ids[:50]) if source_document_ids else None,
                "source_type": doc.source_type if doc else None,
                "title": doc.title if doc else None,
                "local_metadata": getattr(node, "local_metadata", None),
            }
            if sub_scores is not None:
                entry["sub_scores"] = sub_scores[i]
            layer_information.append(entry)
        return layer_information

    # ------------------------------------------------------------------------ hybrid score -------
    def _relaxation_plan(self, applied: List[str]) -> List[Tuple[str, Dict]]:
        """Ordered relaxations for the hard filter, each a (label, state-update) pair."""
        base_tol = int(self.conf.get("hard_time_tolerance_days") or 14)
        plan: List[Tuple[str, Dict]] = []
        if "time" in applied:
            plan.append(("widen_time", {"time_tol": base_tol * 4}))
            plan.append(("drop_time", {"drop": "time"}))
        if "source_type" in applied:
            plan.append(("drop_source_type", {"drop": "source_type"}))
        if "ticket_keys" in applied:
            plan.append(("drop_ticket_keys", {"drop": "ticket_keys"}))
        return plan

    def _retrieve_hybrid_score(self, query: str, query_embedding, extras: Optional[Dict],
                               question_type: Optional[str]) -> Tuple[List[str], List[Dict], float, Dict[str, float]]:
        from .query.candidates import build_candidate_pool
        from .query.scoring import level_score, metadata_match, select_with_mmr

        start_time = time.time()
        self._ensure_hybrid_structures()
        conf = self.conf
        top_k = conf["rerank_top_k"] if conf.get("rerank_top_k") is not None else conf["tree_top_k"]

        # 1) query understanding
        t0 = time.time()
        constraints = self.query_understanding.parse(query)
        parse_time = time.time() - t0

        # 2) dense similarities (whole tree) + BM25 vector (leaves) + traversal seeds
        t0 = time.time()
        if query_embedding is None:
            query_embedding = self.embed(query)
        sims = self._dense_index.sims(query_embedding)
        seed_nodes = None
        if self.candidate_mode == "traversal":
            layer_nodes = [self.tree.all_nodes[idx] for idx in self.tree.layer_to_node_indices[conf["start_layer"]]]
            _, _, _, visited = self._traverse(layer_nodes, query, conf["start_layer"], query_embedding=query_embedding)
            seed_nodes = list(dict.fromkeys(idx for idx, _ in visited))
        dense_time = time.time() - t0
        t0 = time.time()
        scorer = self.sparse_scorer
        bm25_vector = scorer.scores(query) if scorer is not None else None
        sparse_time = time.time() - t0

        # 3) candidate pool, with the hard filter and its relaxation loop
        t0 = time.time()
        use_filter = bool(conf.get("metadata_filter")) and constraints.has_constraints()
        hard_fields = list(conf.get("hard_filter_fields") or [])
        time_tol = int(conf.get("hard_time_tolerance_days") or 14)
        relaxations: List[str] = []
        applied: List[str] = []
        node_mask = None
        if use_filter:
            node_mask, applied = self._meta_index.hard_mask(constraints, hard_fields, time_tol)
        plan = self._relaxation_plan(applied) if use_filter else []
        while True:
            pool = build_candidate_pool(
                self.tree, self._dense_index, sims, bm25_vector, self.tree_node_index_to_layer,
                self._relations, self._leaf_cache,
                int(conf.get("candidate_dense_top_n") or 100), int(conf.get("candidate_sparse_top_n") or 100),
                node_mask=node_mask, seed_nodes=seed_nodes,
            )
            if not use_filter or len(pool) >= top_k or not plan:
                break
            label, update = plan.pop(0)
            relaxations.append(label)
            if "time_tol" in update:
                time_tol = update["time_tol"]
            if "drop" in update:
                hard_fields = [f for f in hard_fields if f != update["drop"]]
            node_mask, applied = self._meta_index.hard_mask(constraints, hard_fields, time_tol)
            if not applied:
                node_mask, use_filter = None, False

        # 4) metadata / level terms and greedy selection
        field_weights = conf.get("metadata_field_weights") or None
        tol_days = int(conf.get("time_tolerance_days") or 0)
        level_pref = conf.get("level_preference") or None
        candidates = list(pool.values())
        for cand in candidates:
            cand.metadata, cand.meta_fields = metadata_match(constraints, self._meta_index.view(cand.node_index),
                                                             field_weights, tol_days)
            cand.level = level_score(question_type, cand.layer, level_pref)
        selected = select_with_mmr(candidates, conf.get("score_weights"), top_k, self._relations,
                                   str(conf.get("score_norm") or "minmax"))
        score_time = time.time() - t0

        final_nodes = [self.tree.all_nodes[c.node_index] for c, _, _ in selected]
        scores = [s for _, s, _ in selected]
        sub_scores = [sub for _, _, sub in selected]
        context = self._decorate_context(get_text_list(final_nodes), final_nodes)
        end_time = time.time()
        layer_information = self._layer_information(final_nodes, scores, sub_scores)

        if extras is not None:
            extras["query_parse"] = constraints.to_dict()
            extras["question_type"] = question_type
            extras["filters_applied"] = applied if use_filter else []
            extras["relaxations"] = relaxations
            extras["pool_size"] = len(pool)
            extras["candidates"] = [
                {"node_index": c.node_index, "layer": c.layer, "origin": c.origin, "dense": float(c.dense),
                 "sparse": float(c.sparse), "metadata": float(c.metadata), "level": float(c.level),
                 "meta_fields": c.meta_fields,
                 "document_ids": self._meta_index.view(c.node_index).document_ids[:5]}
                for c in candidates
            ]
            extras["selected"] = [c.node_index for c, _, _ in selected]

        return (context, layer_information, end_time - start_time,
                {'tree': dense_time, 'sparse': sparse_time, 'rerank': 0.0, 'parse': parse_time, 'score': score_time})

    # ------------------------------------------------------------------------------- BM25 --------
    def hybrid_index(self, docs: List[str]) -> None:
        '''Build a sparse keyword index with BM25. '''
        if hasattr(self.hybrid_search_model, "vocab_dict"):
            return
        
        if self.conf["save_dir"] is not None:
            hybrid_save_dir = os.path.join(
                self.conf["save_dir"],
                get_sparse_save_name(self.conf),
            )
            if self.conf["force_sparse_index_from_scratch"] and os.path.exists(hybrid_save_dir):
                shutil.rmtree(hybrid_save_dir)
            corpus_tokens = bm25s.tokenize(docs, stopwords="en", stemmer=self.stemmer, show_progress=False)
            self.hybrid_search_model.index(corpus_tokens)
            self.hybrid_search_model.save(hybrid_save_dir)
        else:
            corpus_tokens = bm25s.tokenize(docs, stopwords="en", stemmer=self.stemmer, show_progress=False)
            self.hybrid_search_model.index(corpus_tokens)
        self._sparse_scorer = None   # rebuilt lazily with the new model

    def _hybrid_retrieve(self, query: str, top_k: int | str = 5) -> List[int]:
        '''Retrieve chunks from the sparse keyword index. '''
        if self.hybrid_search_model is None:
            raise ValueError("There is no model for hybrid search.")
        elif not hasattr(self.hybrid_search_model, "vocab_dict"):
            raise ValueError("There is no index for hybrid search. Call ``hybrid_index()`` first. ")

        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer, show_progress=False)
        retrieved_node_indices, scores = list(map(lambda x: x[0], 
            self.hybrid_search_model.retrieve(query_tokens, 
                k=min(5 * top_k, self.hybrid_search_model.scores["num_docs"]), 
                sorted=True, 
                show_progress=False
            )
        ))
        
        retrieved_docs = {}
        # Deduplication
        for node_idx in retrieved_node_indices: 
            text = self.tree.all_nodes[node_idx].text
            duplicate_fn = lambda t, db: t in db
            if duplicate_fn(text, retrieved_docs):
                retrieved_docs[text] = min(int(node_idx), retrieved_docs[text])
            else:
                retrieved_docs[text] = int(node_idx)

        return list(retrieved_docs.values())[:top_k]
