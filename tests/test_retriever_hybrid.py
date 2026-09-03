"""TreeRetriever in hybrid_score mode on the 4-leaf fixture tree (offline hash embeddings + BM25)."""
import numpy as np
import pytest

from conf import read_config
from src.metadata import aggregate_tree_metadata
from src.model.fake import HashEmbeddingModel
from src.tree_retriever import TreeRetriever
from tests.test_aggregate import build_tree

TEXTS = {
    0: "alice tan posted in eng about the apollo launch checklist",
    1: "slack thread on apollo rollout and the gpu burst incident",
    2: "jira ticket ENG-1 bob lim investigates apollo gpu driver stalls",
    3: "resolution notes for ENG-1 driver upgrade fixed the stalls",
    4: "summary of slack discussion about apollo launch and gpu burst",
    5: "summary of jira ticket ENG-1 apollo gpu driver stalls resolution",
    6: "root summary apollo launch and ENG-1 gpu incident across slack and jira",
}


def make_retriever(**overrides):
    tree, registry = build_tree()
    registry["doc_b"].metadata.ticket_keys = ["ENG-1"]
    if overrides.pop("no_documents", False):
        registry = None
    else:
        aggregate_tree_metadata(tree, registry)
    embed = HashEmbeddingModel()
    for index, text in TEXTS.items():
        tree.all_nodes[index].text = text
        tree.all_nodes[index].embeddings = embed.embed(text)
    conf = read_config(None)
    conf.update({
        "embed_model": embed, "hybrid_search": True, "save_dir": None, "tree_top_k": 2, "sparse_top_k": 2,
        "rerank_top_k": 3, "query_understanding": "rules", "context_metadata_header": True,
        "candidate_dense_top_n": 4, "candidate_sparse_top_n": 2,
    })
    conf.update(overrides)
    retriever = TreeRetriever(conf, tree)
    retriever.hybrid_index([TEXTS[i] for i in range(4)])
    return retriever, tree


def test_legacy_mode_unchanged():
    retriever, tree = make_retriever(retrieve_mode="legacy")
    context, info, _, times = retriever.retrieve("apollo gpu driver stalls")
    assert 2 <= len(context) <= 3 and all(c.startswith("[doc: ") for c in context)
    assert all("sub_scores" not in entry for entry in info)
    assert all(entry["layer_number"] == 0 for entry in info)
    assert set(times) == {"tree", "sparse", "rerank"}
    assert retriever._dense_index is None                      # nothing hybrid was built


def test_collapsed_pool_and_extras():
    retriever, tree = make_retriever(retrieve_mode="hybrid_score")
    extras = {}
    context, info, _, times = retriever.retrieve("what did Bob Lim find about ENG-1 gpu driver stalls in the jira ticket",
                                                 extras=extras, question_type="basic")
    assert len(info) == 3 and {"parse", "score"} <= set(times)
    assert extras["query_parse"]["ticket_keys"] == ["ENG-1"] and extras["query_parse"]["source_types"] == ["jira"]
    assert extras["query_parse"]["people"] == ["Bob Lim"]
    layers = {c["layer"] for c in extras["candidates"]}
    assert 0 in layers and layers - {0}                        # abstracts are in the pool
    origins = {c["origin"] for c in extras["candidates"]}
    assert origins & {"sparse", "both", "sparse-parent"}
    # abstract S_BM25 is propagated from its leaves
    by_index = {c["node_index"]: c for c in extras["candidates"]}
    if 5 in by_index and 2 in by_index:
        assert by_index[5]["sparse"] >= max(by_index[2]["sparse"], by_index.get(3, {"sparse": 0})["sparse"]) - 1e-6
    # doc_b (jira, ENG-1, Bob Lim) outranks doc_a on metadata
    meta = {c["node_index"]: c["metadata"] for c in extras["candidates"]}
    assert meta.get(2, 0) > meta.get(0, 0)
    top = info[0]
    assert top["sub_scores"]["metadata"] > 0 and "meta_fields" in top["sub_scores"]
    assert top["document_id"] == "doc_b" or top["source_document_ids"] == ["doc_b"]
    assert extras["filters_applied"] == [] and extras["relaxations"] == []
    assert len({e["node_index"] for e in info}) == 3         # keyed by node index


def test_level_preference_promotes_abstracts():
    retriever, tree = make_retriever(retrieve_mode="hybrid_score",
                                     score_weights={"alpha": 1, "beta": 0, "gamma": 0, "delta": 2.0, "lambda": 0})
    _, info, _, _ = retriever.retrieve("apollo launch and gpu incident", question_type="high_level")
    assert info[0]["layer_number"] >= 1
    _, info_leaf, _, _ = retriever.retrieve("apollo launch and gpu incident", question_type="basic")
    assert info_leaf[0]["layer_number"] == 0


def test_hard_filter_and_relaxation():
    retriever, tree = make_retriever(retrieve_mode="hybrid_score", metadata_filter=True, rerank_top_k=2)
    extras = {}
    _, info, _, _ = retriever.retrieve("what happened in the jira ticket about apollo", extras=extras)
    assert extras["filters_applied"] == ["source_type"] and extras["relaxations"] == []
    assert all(retriever.metadata_index.view(e["node_index"]).source_types <= {"jira", "slack"} for e in info)
    assert all("slack" not in retriever.metadata_index.view(e["node_index"]).source_types
               or "jira" in retriever.metadata_index.view(e["node_index"]).source_types for e in info)
    # a window nobody matches: time is dropped, then the pool is large enough again
    extras = {}
    _, info, _, _ = retriever.retrieve("apollo status on 2020-01-01 from the jira ticket", extras=extras)
    assert "time" in extras["relaxations"][0] or "time" in extras["relaxations"][1]
    assert len(info) == 2


def test_traversal_candidates():
    retriever, tree = make_retriever(retrieve_mode="hybrid_score", candidate_mode="traversal")
    extras = {}
    _, info, _, _ = retriever.retrieve("apollo gpu driver stalls", extras=extras)
    assert len(info) == 3 and any(c["origin"] == "traversal" for c in extras["candidates"])


def test_tree_without_metadata_falls_back_to_neutral_terms():
    retriever, tree = make_retriever(retrieve_mode="hybrid_score", no_documents=True, context_metadata_header=False)
    extras = {}
    context, info, _, _ = retriever.retrieve("apollo gpu driver stalls in the jira ticket ENG-1", extras=extras)
    assert len(info) == 3 and extras["query_parse"]["method"] == "none"
    assert all(c["metadata"] == 0.0 for c in extras["candidates"])
    assert retriever.query_understanding.mode == "none"


def test_metatext_index_skips_runtime_header():
    retriever, tree = make_retriever(retrieve_mode="hybrid_score", enterprise_chunk_metadata_prefix=True)
    context, _, _, _ = retriever.retrieve("apollo gpu")
    assert not any(c.startswith("[doc: ") or c.startswith("[summary") for c in context)
