"""Stratified dev/test split + DataManager filtering by enterprise_split."""
import json
import os

from experiments.make_splits import make_splits


def test_make_splits_stratified_and_deterministic():
    questions = [{"question_id": f"q{i}", "question_type": "basic" if i < 20 else "rare"} for i in range(26)]
    a = make_splits(questions, dev_frac=0.3, seed=1)
    b = make_splits(questions, dev_frac=0.3, seed=1)
    assert a["dev"] == b["dev"] and not set(a["dev"]) & set(a["test"])
    assert len(a["dev"]) + len(a["test"]) == 26
    assert a["meta"]["by_type"]["basic"]["dev"] == 6 and a["meta"]["by_type"]["rare"]["dev"] == 2
    assert make_splits(questions, dev_frac=0.3, seed=2)["dev"] != a["dev"]


def test_datamanager_split_filter(tmp_path):
    from src import DataManager
    from tests.test_subset import make_dataset
    data_dir = make_dataset(str(tmp_path / "bench"), n_per_type=3)
    all_data = DataManager("enterprise_rag", enterprise_kwargs={
        "data_dir": data_dir, "subset_size": 27, "cache_dir": str(tmp_path / "cache")})
    ids = all_data.question_ids
    split_file = tmp_path / "splits.json"
    split_file.write_text(json.dumps({"dev": ids[:1], "test": ids[1:], "meta": {}}))
    dev = DataManager("enterprise_rag", enterprise_kwargs={
        "data_dir": data_dir, "subset_size": 27, "cache_dir": str(tmp_path / "cache"),
        "split": "dev", "split_file": str(split_file)})
    assert dev.question_ids == ids[:1] and len(dev.all_queries) == 1 and len(dev.gold_doc_ids) == 1
    assert len(dev.documents) == len(all_data.documents)          # the corpus is untouched
    assert dev.subset_stats["split"] == "dev"
    try:
        DataManager("enterprise_rag", enterprise_kwargs={
            "data_dir": data_dir, "subset_size": 27, "cache_dir": str(tmp_path / "cache"),
            "split": "dev", "split_file": str(tmp_path / "missing.json")})
        assert False, "missing split file must raise"
    except FileNotFoundError:
        pass
