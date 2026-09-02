import json
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.enterprise_rag import build_subset, load_enterprise_rag, read_questions, subset_cache_path

TYPES = ("confluence", "fireflies", "github", "gmail", "google_drive", "hubspot", "jira", "linear", "slack")


def make_dataset(root, n_per_type=12):
    os.makedirs(os.path.join(root, "data", "documents"))
    os.makedirs(os.path.join(root, "data", "questions"))
    rows = []
    for t in TYPES:
        for i in range(n_per_type):
            rows.append({"doc_id": f"dsid_{t}_{i:03d}", "source_type": t, "title": f"{t} {i}", "content": f"content {t} {i}"})
    rows.append(dict(rows[0]))  # duplicate doc_id
    pq.write_table(pa.Table.from_pylist(rows), os.path.join(root, "data", "documents", "test.parquet"))
    questions = [
        {"question_id": "qst_0001", "question_type": "basic", "source_types": ["jira"], "question": "q1",
         "expected_doc_ids": ["dsid_jira_003"], "gold_answer": "a1", "answer_facts": ["f1"]},
        {"question_id": "qst_0002", "question_type": "completeness", "source_types": ["slack", "gmail"], "question": "q2",
         "expected_doc_ids": ["dsid_slack_005", "dsid_gmail_007"], "gold_answer": "a2", "answer_facts": ["f1", "f2"]},
        {"question_id": "qst_0003", "question_type": "info_not_found", "source_types": [], "question": "q3",
         "expected_doc_ids": [], "gold_answer": "not found", "answer_facts": []},
    ]
    schema = pa.schema([("question_id", pa.string()), ("question_type", pa.string()),
                        ("source_types", pa.list_(pa.string())), ("question", pa.string()),
                        ("expected_doc_ids", pa.list_(pa.string())), ("gold_answer", pa.string()),
                        ("answer_facts", pa.list_(pa.string()))])
    pq.write_table(pa.Table.from_pylist(questions, schema=schema), os.path.join(root, "data", "questions", "test.parquet"))
    return root


@pytest.fixture
def dataset_dir(tmp_path):
    return make_dataset(str(tmp_path / "bench"))


def test_read_questions(dataset_dir):
    qs = read_questions(dataset_dir)
    assert len(qs) == 3 and qs[1]["expected_doc_ids"] == ["dsid_slack_005", "dsid_gmail_007"]
    assert qs[2]["answer_facts"] == [] and qs[2]["source_types"] == []


def test_build_subset_gold_quota_dedup_determinism(dataset_dir):
    docs, qs, stats = build_subset(dataset_dir, subset_size=30, seed=1)
    ids = [d["doc_id"] for d in docs]
    assert len(ids) == len(set(ids)) == 30
    assert {"dsid_jira_003", "dsid_slack_005", "dsid_gmail_007"} <= set(ids)
    assert stats["num_gold"] == 3 and stats["duplicates_skipped"] == 1 and stats["missing_gold"] == 0
    # 27 distractors over 9 types: 3 each; gold docs do not consume quota
    assert stats["quota"] == {t: 3 for t in TYPES}
    assert stats["per_source_type"] == {"confluence": 3, "fireflies": 3, "github": 3, "gmail": 4, "google_drive": 3,
                                        "hubspot": 3, "jira": 4, "linear": 3, "slack": 4}
    docs2, _, _ = build_subset(dataset_dir, subset_size=30, seed=1)
    assert [d["doc_id"] for d in docs2] == ids
    docs3, _, _ = build_subset(dataset_dir, subset_size=30, seed=2)
    assert [d["doc_id"] for d in docs3] != ids
    assert ids == sorted(ids, key=lambda x: (x.split("_")[1], x))  # sorted by (source_type, doc_id)


def test_build_subset_full_and_too_small(dataset_dir):
    docs, _, stats = build_subset(dataset_dir, subset_size=None)
    assert stats["num_documents"] == 9 * 12 and stats["quota"] == {}
    with pytest.raises(ValueError):
        build_subset(dataset_dir, subset_size=2)


def test_cache_reuse_and_invalidation(dataset_dir, tmp_path):
    cache_dir = str(tmp_path / "cache")
    docs, qs, stats = load_enterprise_rag(dataset_dir, subset_size=20, seed=3, cache_dir=cache_dir, verbose=False)
    path = subset_cache_path(cache_dir, 20, 3)
    assert os.path.exists(path)
    with open(path) as f:
        meta = json.loads(f.readline())["meta"]
    assert meta["subset_size"] == 20 and meta["seed"] == 3
    docs2, qs2, _ = load_enterprise_rag(dataset_dir, subset_size=20, seed=3, cache_dir=cache_dir, verbose=False)
    assert [d["doc_id"] for d in docs2] == [d["doc_id"] for d in docs]
    assert qs2 == qs
    # corrupt the header -> rebuilt
    with open(path, "w") as f:
        f.write('{"meta": {"format": "old"}}\n')
    docs3, _, _ = load_enterprise_rag(dataset_dir, subset_size=20, seed=3, cache_dir=cache_dir, verbose=False)
    assert len(docs3) == 20
