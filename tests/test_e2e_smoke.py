"""End-to-end: DataManager -> chunking -> tree with metadata -> retrieval with provenance (offline models)."""
import os

import pytest

from conf import read_config
from src import RAG, DataManager
from src.dataset import enterprise_kwargs_from_conf, split_dataset
from src.metadata import collect_sources
from src.model.factory import build_model
from tests.test_subset import make_dataset

pytestmark = pytest.mark.slow


def enrich(rows_root):
    """Give the synthetic corpus realistic per-source content so the parsers have something to find."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = os.path.join(rows_root, "data", "documents", "test.parquet")
    table = pq.read_table(path).to_pylist()
    fixtures = {}
    for name in os.listdir(os.path.join("tests", "fixtures")):
        with open(os.path.join("tests", "fixtures", name), encoding="utf-8") as f:
            fixtures[name.split(".")[0].split("_")[0]] = f.read()
    for i, row in enumerate(table):
        base = fixtures.get(row["source_type"], fixtures["confluence"])
        row["content"] = base + f"\n\nUnique filler sentence number {i} about topic {row['doc_id']}. " * 3
    pq.write_table(pa.Table.from_pylist(table), path)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("bench")
    data_dir = make_dataset(str(root / "bench"), n_per_type=8)
    enrich(data_dir)
    conf = read_config("enterprise_rag_smoke")
    conf.update({
        "enterprise_data_dir": data_dir,
        "enterprise_subset_size": 30,
        "enterprise_subset_cache_dir": str(root / "cache"),
        "save_dir": None,
        "test_samples": -1,
        "tree_top_k": 3, "sparse_top_k": 3, "rerank_top_k": 3,
        "max_tokens_per_chunk": 60,
    })
    data = DataManager("enterprise_rag", data_dir=conf["data_dir"], test_samples=conf["test_samples"],
                       enterprise_kwargs=enterprise_kwargs_from_conf(conf))
    for task in ("embed", "abs", "qa"):
        conf[f"{task}_model"] = build_model(conf[f"{task}_name"], task, conf)
    split_dataset(data, conf)
    rag = RAG(conf)
    rag.add_documents(data)
    rag.build_vocab(data)
    return conf, data, rag


def test_dataset_objects(built):
    conf, data, rag = built
    assert len(data.documents) == 30 and len(data.document_registry) == 30
    assert data.all_text_ids == [d.document_id for d in data.documents]
    assert data.gold_doc_ids[0] == ["dsid_jira_003"] and data.gold_doc_ids[2] == []
    assert data.question_types[2] == "info_not_found"
    assert data.gold_answers[0] == {"a1"}
    assert len(data.chunk_local_metadata) == 30
    assert all(chunk.startswith(doc.title + "\n") for doc, chunks in zip(data.documents, data.all_passages) for chunk in chunks)
    gmail = next(d for d in data.documents if d.source_type == "gmail")
    assert gmail.metadata.authors and gmail.metadata.created_at


def test_tree_metadata(built):
    conf, data, rag = built
    tree = rag.tree
    assert tree.documents is data.document_registry
    leaves = [tree.all_nodes[i] for i in tree.layer_to_node_indices[0]]
    assert all(leaf.document_id in tree.documents for leaf in leaves)
    assert all(isinstance(leaf.local_metadata, dict) for leaf in leaves)
    for i in tree.layer_to_node_indices[1]:
        node = tree.all_nodes[i]
        assert node.source_document_ids and len(node.source_document_ids) == 1   # preset chunks: one doc per layer-1 node
        assert node.aggregated_metadata["num_chunks"] == len(node.children)
    root = tree.all_nodes[tree.layer_to_node_indices[max(tree.layer_to_node_indices)][0]]
    assert len(root.source_document_ids) == 30
    assert set(root.aggregated_metadata["source_types"]) == {d.source_type for d in data.documents}
    for doc in tree.documents.values():
        assert doc.chunk_ids and all(tree.all_nodes[c].document_id == doc.document_id for c in doc.chunk_ids)
    # BM25 row position == leaf index coupling
    assert data.get_documents()[5] == tree.all_nodes[5].text


def test_retrieval_provenance(built):
    conf, data, rag = built
    context, info = rag.retrieve("private upgrade rollback audit log", 0)
    assert context and all(c.startswith("[doc: dsid_") for c in context)
    assert all(entry["document_id"] in rag.tree.documents for entry in info)
    assert all(entry["source_type"] and entry["title"] for entry in info)
    sources = collect_sources(rag.tree, {e["node_index"]: e["score"] for e in info})
    ids = [s["document_id"] for s in sources]
    assert len(ids) == len(set(ids)) and set(ids) <= set(rag.tree.documents)
