"""
Stratified dev / test split of the EnterpriseRAG-Bench questions (by question_type).

    python experiments/make_splits.py [--dev-frac 0.3] [--seed 42] [--out conf/enterprise_splits_s42.json]

The split is keyed by question_id, so it is independent of the document subset. Weights of the
hybrid score are tuned on "dev" and reported on "test" (conf: enterprise_split, enterprise_split_file).
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def make_splits(questions, dev_frac: float = 0.3, seed: int = 42):
    by_type = defaultdict(list)
    for q in questions:
        by_type[q.get("question_type") or "unknown"].append(q["question_id"])
    rng = random.Random(seed)
    dev, test = [], []
    for qtype in sorted(by_type):
        ids = sorted(by_type[qtype])
        rng.shuffle(ids)
        n_dev = max(1, int(round(len(ids) * dev_frac))) if len(ids) > 1 else 0
        dev.extend(ids[:n_dev])
        test.extend(ids[n_dev:])
    return {"dev": sorted(dev), "test": sorted(test),
            "meta": {"dev_frac": dev_frac, "seed": seed, "n_dev": len(dev), "n_test": len(test),
                     "by_type": {t: {"dev": sum(1 for i in ids if i in set(dev)), "total": len(ids)}
                                 for t, ids in sorted(by_type.items())}}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data/enterpriseRAG-Bench")
    parser.add_argument("--dev-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from src.enterprise_rag import read_questions
    questions = read_questions(args.data_dir)
    splits = make_splits(questions, args.dev_frac, args.seed)
    out = args.out or os.path.join(ROOT, "conf", f"enterprise_splits_s{args.seed}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=1)
    print(f"wrote {out}: {splits['meta']}")


if __name__ == "__main__":
    main()
