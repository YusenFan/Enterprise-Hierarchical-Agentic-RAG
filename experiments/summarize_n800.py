"""
Merge the retrieval-only arm tables (run_arms.py) with the QA + judge results (qa.py / eval.py,
re-evaluated here from the saved answer files) into one markdown summary.

    python experiments/summarize_n800.py --config enterprise_rag --enterprise_subset_size 800 --split test \
        --exp-dirs output/experiments/n800_test output/experiments/n800_test_b --qa-tags A0,C0,C3,D,Cbest,B,BC3
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.makedirs(os.path.join(ROOT, "log"), exist_ok=True)

from conf import apply_config_overrides, read_config  # noqa: E402
from src import DataManager, Evaluator  # noqa: E402
from src.dataset import enterprise_kwargs_from_conf  # noqa: E402
from src.model.factory import build_model  # noqa: E402
from src.utils import load_answers  # noqa: E402

RET_COLS = ["DocRecall@1", "DocRecall@5", "DocRecall@10", "DocMRR", "DocNDCG@10", "InvalidExtraDocs@5"]
QA_COLS = ["JudgeOverall", "JudgeCorrectness", "JudgeCompleteness", "F1", "DocRecall@5", "DocMRR"]


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join("" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--exp-dirs", nargs="*", default=[])
    parser.add_argument("--qa-tags", default="")
    parser.add_argument("--out", default=None)
    args, unknown = parser.parse_known_args()
    conf = apply_config_overrides(read_config(args.config), unknown)
    conf["enterprise_split"] = args.split

    sections = []
    # ---- retrieval-only arms
    arms = {}
    types = set()
    for d in args.exp_dirs:
        path = os.path.join(d, "arms.json")
        if not os.path.exists(path):
            continue
        data = json.load(open(path))
        for arm, entry in data["arms"].items():
            if arm == "grid_best":
                continue
            arms[arm] = entry
            types |= set(entry["metrics"].get("DocRecall_by_type", {}).keys())
        if data.get("star_weights"):
            sections.append(f"Dev-selected weights (`*`): `{data['star_weights']}`")
    if arms:
        types = sorted(types)
        header = ["arm"] + RET_COLS + [f"R@5 {t}" for t in types]
        rows = []
        for arm, entry in arms.items():
            m = entry["metrics"]
            rows.append([arm] + [m.get(k) for k in RET_COLS] + [m.get("DocRecall_by_type", {}).get(t, {}).get("DocRecall@5") for t in types])
        sections.append(f"### Retrieval only ({args.split} split, first retrieval, n={data['n_questions']})\n\n" + md_table(header, rows))
        for d in args.exp_dirs:
            p = os.path.join(d, "field_ablation.md")
            if os.path.exists(p):
                sections.append(f"### S_metadata field ablation ({os.path.basename(d)})\n\n" + open(p).read().strip())
                break

    # ---- QA + judge
    tags = [t for t in args.qa_tags.split(",") if t]
    if tags:
        data = DataManager(conf["dataset"], data_dir=conf["data_dir"], test_samples=conf["test_samples"],
                           enterprise_kwargs=enterprise_kwargs_from_conf(conf))
        top_k = conf["rerank_top_k"] if conf.get("rerank_top_k") is not None else conf["tree_top_k"]
        rows, by_type_rows = [], []
        qtypes = sorted(set(t for t in data.question_types if t))
        for tag in tags:
            c = dict(conf, run_tag=tag)
            results = load_answers(c)
            if not results:
                rows.append([tag] + [None] * len(QA_COLS))
                continue
            judge = build_model(conf["judge_name"], "judge", conf) if conf.get("judge_name") else None
            ev = Evaluator(data=data, top_k_nodes_per_layer=top_k, judge_model=judge,
                           judge_cache_path=os.path.join(conf["save_dir"], "results", f"{conf['config']}_{tag}_judge.json"))
            scores = ev.evaluate(answers=results.get("answers"), retrieved_doc_ids=results.get("retrieved_doc_ids"),
                                 retrieved_doc_ids_leaf_only=results.get("retrieved_doc_ids_leaf_only"),
                                 metrics=["llmjudge", "f1", "docrecall", "docmrr"])
            rows.append([tag] + [scores.get(k) for k in QA_COLS])
            jt = scores.get("Judge_by_type", {})
            by_type_rows.append([tag] + [jt.get(t, {}).get("JudgeOverall") for t in qtypes])
        sections.append(f"### QA + LLM judge ({args.split} split, agentic QA with max_retrieval_time=1)\n\n"
                        + md_table(["arm"] + QA_COLS, rows)
                        + "\n\nJudgeOverall by question type:\n\n" + md_table(["arm"] + qtypes, by_type_rows))

    text = f"# EnterpriseRAG-Bench n800 — metadata-aware query arms ({args.split})\n\n" + "\n\n".join(sections) + "\n"
    out = args.out or os.path.join(conf["save_dir"], "experiments", f"n800_summary_{args.split}.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
