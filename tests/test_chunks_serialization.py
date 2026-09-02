import pickle

from src.metadata import aggregate_tree_metadata
from src.tree_builder.chunks import _deserialize_node, load_tree_chunks, save_tree_chunks
from tests.test_aggregate import build_tree


def test_round_trip_with_documents(tmp_path):
    tree, registry = build_tree()
    aggregate_tree_metadata(tree, registry)
    path = str(tmp_path / "tree")
    save_tree_chunks(tree, path, chunk_size=3)
    loaded = load_tree_chunks(path)
    assert set(loaded.all_nodes) == set(tree.all_nodes)
    assert loaded.all_nodes[0].document_id == "doc_a"
    assert loaded.all_nodes[0].local_metadata == {"section": "s0"}
    assert loaded.all_nodes[6].source_document_ids == ["doc_a", "doc_b"]
    assert loaded.all_nodes[6].aggregated_metadata["num_documents"] == 2
    assert set(loaded.documents) == {"doc_a", "doc_b"}
    assert loaded.documents["doc_b"].metadata.authors == ["Bob Lim", "Alice Tan"]
    assert loaded.documents["doc_a"].chunk_ids == [0, 1]


def test_v1_node_dict_without_metadata_fields():
    node = _deserialize_node({"text": "t", "index": 3, "document_index": 0, "chunk_index": 0,
                              "children": [], "embeddings": None})
    assert node.document_id is None and node.aggregated_metadata is None


def test_pickle_round_trip_keeps_metadata():
    tree, registry = build_tree()
    aggregate_tree_metadata(tree, registry)
    loaded = pickle.loads(pickle.dumps(tree))
    assert loaded.all_nodes[4].source_refs == [{"document_id": "doc_a", "chunk_ids": [0, 1]}]
    assert loaded.documents["doc_a"].title == "title doc_a"
