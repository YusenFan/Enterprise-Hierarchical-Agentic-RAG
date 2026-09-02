import numpy as np

from src.metadata import Document, DocumentMetadata, TimeRange, aggregate_tree_metadata, collect_sources, format_context_header
from src.metadata.aggregate import aggregate_documents, node_document_ids
from src.utils import Node, Tree


def _doc(doc_id, source_type, authors, start, end, projects=(), channel=None):
    return Document(doc_id, f"title {doc_id}", source_type, DocumentMetadata(
        authors=list(authors), participants=list(authors), projects=list(projects), channel=channel,
        created_at=start, updated_at=end, time_range=TimeRange(start, end), entities=["ENG-1"]))


def build_tree():
    """4 leaves (2 docs) -> 2 layer-1 abstracts (one per doc) -> root."""
    emb = np.ones(4, dtype=np.float32)
    nodes = {
        0: Node("c0", 0, 0, 0, set(), emb, document_id="doc_a", local_metadata={"section": "s0"}),
        1: Node("c1", 1, 0, 1, set(), emb, document_id="doc_a"),
        2: Node("c2", 2, 1, 0, set(), emb, document_id="doc_b"),
        3: Node("c3", 3, 1, 1, set(), emb, document_id="doc_b"),
        4: Node("abs a", 4, -1, -1, {0, 1}, emb),
        5: Node("abs b", 5, -1, -1, {2, 3}, emb),
        6: Node("root", 6, -1, -1, {4, 5}, emb),
    }
    layers = {0: [0, 1, 2, 3], 1: [4, 5], 2: [6]}
    tree = Tree(nodes, {6: nodes[6]}, {i: nodes[i] for i in range(4)}, layers)
    registry = {
        "doc_a": _doc("doc_a", "slack", ["Alice Tan"], "2026-08-10", "2026-08-11", ["Apollo"], channel="eng"),
        "doc_b": _doc("doc_b", "jira", ["Bob Lim", "Alice Tan"], "2026-08-12", "2026-08-18", ["Apollo"]),
    }
    return tree, registry


def test_node_defaults_backward_compatible():
    node = Node("t", 0, 0, 0, set(), None)
    assert node.document_id is None and node.local_metadata is None
    assert node.source_refs is None and node.aggregated_metadata is None
    assert node.is_leaf
    assert Tree({0: node}, {0: node}, {0: node}, {0: [0]}).documents is None


def test_aggregate_tree_metadata():
    tree, registry = build_tree()
    aggregate_tree_metadata(tree, registry)
    assert tree.documents is registry
    assert registry["doc_a"].chunk_ids == [0, 1] and registry["doc_b"].chunk_ids == [2, 3]
    a = tree.all_nodes[4]
    assert a.source_refs == [{"document_id": "doc_a", "chunk_ids": [0, 1]}]
    assert a.source_document_ids == ["doc_a"]
    assert a.aggregated_metadata["num_documents"] == 1 and a.aggregated_metadata["num_chunks"] == 2
    assert a.aggregated_metadata["source_types"] == ["slack"]
    assert a.aggregated_metadata["channels"] == ["eng"]
    root = tree.all_nodes[6]
    assert root.source_document_ids == ["doc_a", "doc_b"]
    assert root.source_refs == [{"document_id": "doc_a", "chunk_ids": [0, 1]},
                                {"document_id": "doc_b", "chunk_ids": [2, 3]}]
    agg = root.aggregated_metadata
    assert agg["num_documents"] == 2 and agg["num_chunks"] == 4
    assert agg["source_types"] == ["jira", "slack"]
    assert agg["authors"] == ["Alice Tan", "Bob Lim"]          # Alice appears in both docs -> first
    assert agg["projects"] == ["Apollo"]
    assert agg["time_range"] == {"start": "2026-08-10", "end": "2026-08-18"}
    assert agg["latest_updated_at"] == "2026-08-18"
    assert agg["source_authority"] == 3                        # jira (3) > slack (1)
    # leaves untouched
    assert tree.all_nodes[0].source_refs is None and node_document_ids(tree.all_nodes[0]) == ["doc_a"]


def test_aggregate_documents_empty():
    agg = aggregate_documents([])
    assert agg["num_documents"] == 0 and agg["time_range"] == {"start": None, "end": None}
    assert agg["source_authority"] == 0


def test_format_context_header_and_sources():
    tree, registry = build_tree()
    aggregate_tree_metadata(tree, registry)
    leaf_header = format_context_header(tree.all_nodes[0], registry["doc_a"])
    assert leaf_header == "[doc: doc_a | slack | title doc_a | 2026-08-10 | Alice Tan]"
    assert format_context_header(tree.all_nodes[2]) == "[doc: doc_b]"
    assert format_context_header(tree.all_nodes[6]) == "[summary | 2 docs | jira, slack | 2026-08-10..2026-08-18]"
    sources = collect_sources(tree, {2: 0.4, 0: 0.9, 1: 0.5, 6: 0.1})
    assert [s["document_id"] for s in sources] == ["doc_a", "doc_b"]
    assert sources[0] == {"document_id": "doc_a", "source_type": "slack", "title": "title doc_a", "best_score": 0.9}
    assert collect_sources(tree, {6: 0.2})[1]["document_id"] == "doc_b"
