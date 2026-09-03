"""
Retrieval-only experiment runner for the metadata-aware arms (no answer LLM, no judge).

    python experiments/run_arms.py --config enterprise_rag --split dev --grid
    python experiments/run_arms.py --config enterprise_rag --split test --weights-from output/experiments/enterprise_rag_dev/grid.json
    python experiments/run_arms.py --config enterprise_rag --arms A0,C0,C3 --tag quick --test_samples 50

Loads the tree / BM25 index once, runs the *first* retrieval of every question for each arm (same
query embeddings, shared dense matrix), computes DocRecall / DocMRR / DocNDCG / InvalidExtraDocs
(overall and per question_type) and writes a comparison table. `--grid` re-scores the cached
candidate pools of the reference soft arm for every weight combination (pure numpy, seconds).
Any other "--key value" pair is a config override, exactly like qa.py.
"""
import argparse
import csv
import itertools
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.makedirs(os.path.join(ROOT, "log"), exist_ok=True)

from tqdm import tqdm  # noqa: E402

from conf import apply_config_overrides, read_config  # noqa: E402
from src import RAG, DataManager, Evaluator  # noqa: E402
from src.dataset import enterprise_kwargs_from_conf  # noqa: E402
from src.model.factory import build_model  # noqa: E402
from src.query.credit import document_credit, merge_node_scores  # noqa: E402
from src.query.scoring import Candidate, select_with_mmr  # noqa: E402
from src.tree_retriever import TreeRetriever  # noqa: E402
from src.utils import get_sparse_save_name, get_tree_save_name, is_bucketed_tree, result_stem, save_answers  # noqa: E402

STAR = "*"   # resolved from --weights-from (or the config defaults)
SOFT = {"retrieve_mode": "hybrid_score", "candidate_mode": "collapsed", "metadata_filter": False}
ARMS: Dict[str, Dict] = {
    "A0":  {"retrieve_mode": "legacy", "query_understanding": "none"},
    "C0":  {**SOFT, "query_understanding": "none", "score_weights": {"gamma": 0.0, "delta": 0.0, "lambda": 0.0}},
    "C1":  {**SOFT, "query_understanding": "llm", "score_weights": {"gamma": STAR, "delta": 0.0, "lambda": 0.0}},
    "C1r": {**SOFT, "query_understanding": "rules", "score_weights": {"gamma": STAR, "delta": 0.0, "lambda": 0.0}},
    "C2":  {**SOFT, "query_understanding": "llm", "score_weights": {"gamma": STAR, "delta": STAR, "lambda": 0.0}},
    "C3":  {**SOFT, "query_understanding": "llm", "score_weights": {"gamma": STAR, "delta": STAR, "lambda": STAR}},
    "C3t": {**SOFT, "candidate_mode": "traversal", "query_understanding": "llm",
            "score_weights": {"gamma": STAR, "delta": STAR, "lambda": STAR}},
    "D":   {**SOFT, "metadata_filter": True, "query_understanding": "llm",
            "score_weights": {"gamma": STAR, "delta": STAR, "lambda": STAR}},
    "B":   {"retrieve_mode": "legacy", "query_understanding": "none", "enterprise_chunk_metadata_prefix": True},
    "BC3": {**SOFT, "query_understanding": "llm", "enterprise_chunk_metadata_prefix": True,
            "score_weights": {"gamma": STAR, "delta": STAR, "lambda": STAR}},
}
GRID_REFERENCE_ARM = "C3"
GRID = {"beta": [0.25, 0.5, 1.0], "gamma": [0.0, 0.25, 0.5, 1.0], "delta": [0.0, 0.3, 0.6], "lambda": [0.0, 0.3, 0.6]}
TABLE_COLUMNS = ["DocRecall@1", "DocRecall@5", "DocRecall@10", "DocMRR", "DocNDCG@5", "DocNDCG@10",
                 "InvalidExtraDocs@5", "DocRecallLeaf@5"]
TYPE_COLUMN = "DocRecall@5"


def resolve_weights(base: Dict[str, float], overrides: Dict, star: Dict[str, float]) -> Dict[str, float]:
    weights = dict(base)
    for key, value in (overrides or {}).items():
        weights[key] = float(star.get(key, base.get(key, 0.0))) if value == STAR else float(value)
    return weights


def arm_conf(conf: Dict, arm: str, star: Dict[str, float]) -> Dict:
    c = dict(conf)
    for key, value in ARMS[arm].items():
        if key == "score_weights":
            c[key] = resolve_weights(conf["score_weights"], value, star)
        else:
            c[key] = value
    c["run_tag"] = arm
    return c


def index_available(conf: Dict) -> bool:
    if conf.get("save_dir") is None:
        return False
    tree = os.path.join(conf["save_dir"], get_tree_save_name(conf))
    if not os.path.exists(tree):
        return False
    if conf.get("hybrid_search"):
        return os.path.exists(os.path.join(conf["save_dir"], get_sparse_save_name(conf), "params.index.json"))
    return True


def load_tree(conf: Dict, cache: Dict[str, object]):
    path = os.path.join(conf["save_dir"], get_tree_save_name(conf))
    if path not in cache:
        tqdm.write(f'Loading tree "{path}"...')
        rag = RAG(conf)
        rag.load("tree", path)
        cache[path] = rag.tree
    return cache[path]


def run_arm(arm: str, conf: Dict, data: DataManager, tree, embeddings: List, structures: Dict, workers: int
            ) -> Dict:
    retriever = TreeRetriever(dict(conf), tree)
    key = id(tree)
    if conf.get("retrieve_mode") == "hybrid_score":
        if key in structures:
            (retriever._dense_index, retriever._meta_index, retriever._relations,
             retriever._leaf_cache) = structures[key]
            retriever._ensure_hybrid_structures()
        else:
            retriever._ensure_hybrid_structures()
            structures[key] = (retriever._dense_index, retriever._meta_index, retriever._relations,
                               retriever._leaf_cache)
    hybrid = conf.get("retrieve_mode") == "hybrid_score"
    top_k = conf["rerank_top_k"] if conf.get("rerank_top_k") is not None else conf["tree_top_k"]
    if not hybrid:
        top_k = min(conf["tree_top_k"], top_k)
    cap = conf.get("source_max_abstract_docs")

    def one(i: int):
        extras: Dict = {}
        qtype = data.question_types[i] if data.question_types else None
        t0 = time.time()
        _, info, _, times = retriever.retrieve(data.all_queries[i], query_embedding=embeddings[i],
                                               extras=extras, question_type=qtype)
        scores = merge_node_scores([info], top_k, all_layers=hybrid)
        sources, leaf_only = document_credit(tree, scores, cap)
        return i, {
            "retrieved_doc_ids": [s["document_id"] for s in sources],
            "retrieved_doc_ids_leaf_only": [s["document_id"] for s in leaf_only],
            "nodes": [(e["node_index"], e["layer_number"], round(float(e["score"]), 5)) for e in info],
            "query_parse": extras.get("query_parse"),
            "filters_applied": extras.get("filters_applied"),
            "relaxations": extras.get("relaxations"),
            "pool_size": extras.get("pool_size"),
            "candidates": extras.get("candidates"),
            "time": time.time() - t0,
            "times": times,
        }

    per_question: Dict[int, Dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(one, i) for i in range(len(data.all_queries))]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"arm {arm}", leave=False):
            i, row = future.result()
            per_question[i] = row
    rows = [per_question[i] for i in range(len(data.all_queries))]
    stats = {}
    if retriever.query_understanding is not None:
        stats = dict(retriever.query_understanding.stats)
    return {"rows": rows, "retriever": retriever, "parse_stats": stats,
            "avg_time": sum(r["time"] for r in rows) / max(len(rows), 1)}


def evaluate_rows(evaluator: Evaluator, rows: List[Dict]) -> Dict:
    return evaluator.evaluate(
        retrieved_doc_ids=[r["retrieved_doc_ids"] for r in rows],
        retrieved_doc_ids_leaf_only=[r["retrieved_doc_ids_leaf_only"] for r in rows],
        metrics=["docrecall", "docmrr", "docndcg", "extradocs"],
    )


def grid_search(evaluator: Evaluator, conf: Dict, result: Dict, tree, top_k: int) -> Dict:
    """Re-score the cached candidate pools of the reference arm for every weight combination."""
    retriever = result["retriever"]
    relations = retriever.relations
    pools = []
    for row in result["rows"]:
        pools.append([Candidate(c["node_index"], c["layer"], dense=c["dense"], sparse=c["sparse"],
                                metadata=c["metadata"], level=c["level"]) for c in (row["candidates"] or [])])
    base = dict(conf["score_weights"])
    cap = conf.get("source_max_abstract_docs")
    records = []
    combos = list(itertools.product(*GRID.values()))
    for combo in tqdm(combos, desc="grid", leave=False):
        weights = dict(base, **dict(zip(GRID.keys(), combo)))
        rows = []
        for pool in pools:
            selected = select_with_mmr(pool, weights, top_k, relations, str(conf.get("score_norm") or "minmax"))
            scores = {c.node_index: s for c, s, _ in selected}
            sources, leaf_only = document_credit(tree, scores, cap)
            rows.append({"retrieved_doc_ids": [s["document_id"] for s in sources],
                         "retrieved_doc_ids_leaf_only": [s["document_id"] for s in leaf_only]})
        metrics = evaluate_rows(evaluator, rows)
        records.append({"weights": weights, **{k: metrics.get(k) for k in TABLE_COLUMNS if k in metrics}})
    records.sort(key=lambda r: (r.get("DocRecall@5", 0), r.get("DocNDCG@10", 0), r.get("DocMRR", 0)), reverse=True)
    return {"grid": GRID, "reference_arm": GRID_REFERENCE_ARM, "n_questions": len(pools),
            "best": records[0], "records": records}


FIELDS = ["source_type", "time", "people", "projects", "entities", "ticket_keys", "channels"]


def field_ablation(evaluator: Evaluator, conf: Dict, result: Dict, tree, top_k: int) -> Dict:
    """
    Re-score the cached candidate pools with S_metadata restricted to ONE field at a time (and to
    all-but-one), keeping the arm's weights. Answers "which metadata field helps / hurts?".
    """
    retriever = result["retriever"]
    relations = retriever.relations
    weights = dict(conf["score_weights"])
    field_weights = dict(conf.get("metadata_field_weights") or {})
    cap = conf.get("source_max_abstract_docs")
    norm = str(conf.get("score_norm") or "minmax")

    def restricted(meta_fields: Dict[str, float], keep) -> float:
        items = [(f, v) for f, v in (meta_fields or {}).items() if f in keep]
        if not items:
            return 0.0
        total = sum(field_weights.get(f, 1.0) for f, _ in items)
        return sum(field_weights.get(f, 1.0) * v for f, v in items) / total if total else 0.0

    def run(keep, label):
        rows = []
        n_specified = 0
        for row in result["rows"]:
            pool = []
            any_field = False
            for c in row["candidates"] or []:
                m = restricted(c.get("meta_fields") or {}, keep)
                any_field |= bool(set((c.get("meta_fields") or {}).keys()) & set(keep))
                pool.append(Candidate(c["node_index"], c["layer"], dense=c["dense"], sparse=c["sparse"],
                                      metadata=m, level=c["level"]))
            n_specified += int(any_field)
            selected = select_with_mmr(pool, weights, top_k, relations, norm)
            scores = {c.node_index: sc for c, sc, _ in selected}
            sources, leaf_only = document_credit(tree, scores, cap)
            rows.append({"retrieved_doc_ids": [s["document_id"] for s in sources],
                         "retrieved_doc_ids_leaf_only": [s["document_id"] for s in leaf_only]})
        metrics = evaluate_rows(evaluator, rows)
        return {"label": label, "fields": list(keep), "n_questions_with_field": n_specified,
                **{k: metrics.get(k) for k in TABLE_COLUMNS if k in metrics}}

    records = [run([], "no_metadata"), run(FIELDS, "all_fields")]
    records += [run([f], f"only_{f}") for f in FIELDS]
    records += [run([g for g in FIELDS if g != f], f"without_{f}") for f in FIELDS]
    return {"weights": weights, "records": records}


def write_tables(out_dir: str, summary: Dict[str, Dict], question_types: List[str]) -> str:
    types = sorted(set(t for t in question_types if t))
    header = ["arm"] + TABLE_COLUMNS + [f"{TYPE_COLUMN}:{t}" for t in types] + ["avg_s"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    with open(os.path.join(out_dir, "table.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for arm, entry in summary.items():
            m = entry["metrics"]
            by_type = m.get("DocRecall_by_type", {})
            row = [arm] + [m.get(k, "") for k in TABLE_COLUMNS] + \
                  [by_type.get(t, {}).get(TYPE_COLUMN, "") for t in types] + [round(entry["avg_time"], 3)]
            writer.writerow(row)
            lines.append("| " + " | ".join(str(x) for x in row) + " |")
    table = "\n".join(lines)
    with open(os.path.join(out_dir, "table.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--arms", default=",".join(ARMS.keys()))
    parser.add_argument("--split", default=None, help='"dev" / "test" (overrides enterprise_split)')
    parser.add_argument("--tag", default=None, help="output folder under <save_dir>/experiments (default <config>_<split>)")
    parser.add_argument("--grid", action="store_true", help=f"weight grid on the {GRID_REFERENCE_ARM} candidate pools")
    parser.add_argument("--field-ablation", action="store_true",
                        help=f"S_metadata one-field-at-a-time ablation on the {GRID_REFERENCE_ARM} candidate pools")
    parser.add_argument("--weights-from", default=None, help="grid.json whose best weights replace the '*' weights")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--save-results", action="store_true",
                        help="also write qa.py-compatible results/<config>_<arm>.json files (answers empty)")
    args, unknown = parser.parse_known_args()

    conf = apply_config_overrides(read_config(args.config), unknown)
    if args.split:
        conf["enterprise_split"] = args.split
    star = dict(conf["score_weights"])
    if args.weights_from:
        with open(args.weights_from, "r", encoding="utf-8") as f:
            star.update(json.load(f)["best"]["weights"])
        tqdm.write(f"Weights from {args.weights_from}: {star}")
    tag = args.tag or f'{args.config}_{conf.get("enterprise_split") or "all"}'
    out_dir = os.path.join(conf["save_dir"], "experiments", tag)
    os.makedirs(out_dir, exist_ok=True)

    data = DataManager(dataset_name=conf["dataset"], data_dir=conf["data_dir"], test_samples=conf["test_samples"],
                       enterprise_kwargs=enterprise_kwargs_from_conf(conf))
    tqdm.write(f"{len(data.all_queries)} questions ({conf.get('enterprise_split') or 'all'}), "
               f"{len(data.documents or [])} documents parsed.")
    conf["embed_model"] = build_model(conf["embed_name"], "embed", conf)
    needs_llm = any(ARMS[a].get("query_understanding") == "llm" for a in args.arms.split(","))
    if needs_llm and (conf.get("query_name") or conf.get("qa_name")):
        conf["query_model"] = build_model(conf.get("query_name") or conf["qa_name"], "query", conf)

    tqdm.write("Embedding questions once...")
    embeddings = [None] * len(data.all_queries)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(conf["embed_model"].embed, q): i for i, q in enumerate(data.all_queries)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="embed", leave=False):
            embeddings[futures[future]] = future.result()

    top_k = conf["rerank_top_k"] if conf.get("rerank_top_k") is not None else conf["tree_top_k"]
    evaluator = Evaluator(data=data, top_k_nodes_per_layer=top_k)

    trees: Dict[str, object] = {}
    structures: Dict[int, tuple] = {}
    summary: Dict[str, Dict] = {}
    results: Dict[str, Dict] = {}
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if arm not in ARMS:
            tqdm.write(f'Unknown arm "{arm}" (known: {", ".join(ARMS)}); skipped.')
            continue
        c = arm_conf(conf, arm, star)
        if not index_available(c):
            tqdm.write(f'[{arm}] index "{get_tree_save_name(c)}" not found; skipped.')
            continue
        tree = load_tree(c, trees)
        t0 = time.time()
        result = run_arm(arm, c, data, tree, embeddings, structures, args.workers)
        metrics = evaluate_rows(evaluator, result["rows"])
        results[arm] = result
        summary[arm] = {"overrides": {k: v for k, v in c.items() if k in ARMS[arm] or k == "score_weights"},
                        "metrics": metrics, "avg_time": result["avg_time"], "parse_stats": result["parse_stats"],
                        "wall_time": time.time() - t0}
        tqdm.write(f"[{arm}] " + ", ".join(f"{k}={metrics.get(k)}" for k in TABLE_COLUMNS if k in metrics)
                   + (f" | parse={result['parse_stats']}" if result["parse_stats"] else ""))
        with open(os.path.join(out_dir, f"{arm}.json"), "w", encoding="utf-8") as f:
            json.dump({"arm": arm, "conf": summary[arm]["overrides"], "metrics": metrics,
                       "question_ids": data.question_ids, "question_types": data.question_types,
                       "rows": [{k: v for k, v in r.items() if k != "candidates"} for r in result["rows"]]},
                      f, ensure_ascii=False)
        if args.save_results:
            rows = result["rows"]
            payload = {
                "answers": [""] * len(rows),
                "retrieved_docs": [[] for _ in rows],
                "sources": [[{"document_id": d} for d in r["retrieved_doc_ids"]] for r in rows],
                "retrieved_doc_ids": [r["retrieved_doc_ids"] for r in rows],
                "retrieved_doc_ids_leaf_only": [r["retrieved_doc_ids_leaf_only"] for r in rows],
                "query_parses": [[r["query_parse"]] if r["query_parse"] else None for r in rows],
                "time": {"tb_time": -1, "tr_time": result["avg_time"], "qa_time": -1},
            }
            save_answers(c, payload, os.path.join(conf["save_dir"], "results"))
            tqdm.write(f'[{arm}] wrote results/{result_stem(c)}.json')

    if args.grid:
        if GRID_REFERENCE_ARM not in results:
            tqdm.write(f"--grid needs arm {GRID_REFERENCE_ARM}; add it to --arms.")
        else:
            c = arm_conf(conf, GRID_REFERENCE_ARM, star)
            grid = grid_search(evaluator, c, results[GRID_REFERENCE_ARM], trees[os.path.join(c["save_dir"], get_tree_save_name(c))], top_k)
            with open(os.path.join(out_dir, "grid.json"), "w", encoding="utf-8") as f:
                json.dump(grid, f, indent=1)
            tqdm.write(f"grid best: {grid['best']}")
            summary["grid_best"] = {"overrides": grid["best"]["weights"], "metrics": grid["best"], "avg_time": 0.0,
                                    "parse_stats": {}, "wall_time": 0.0}

    if args.field_ablation and GRID_REFERENCE_ARM in results:
        c = arm_conf(conf, GRID_REFERENCE_ARM, star)
        abl = field_ablation(evaluator, c, results[GRID_REFERENCE_ARM],
                             trees[os.path.join(c["save_dir"], get_tree_save_name(c))], top_k)
        with open(os.path.join(out_dir, "field_ablation.json"), "w", encoding="utf-8") as f:
            json.dump(abl, f, indent=1)
        cols = ["label", "n_questions_with_field"] + TABLE_COLUMNS[:6]
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in abl["records"]:
            lines.append("| " + " | ".join(str(r.get(k, "")) for k in cols) + " |")
        ablation_table = "\n".join(lines)
        with open(os.path.join(out_dir, "field_ablation.md"), "w", encoding="utf-8") as f:
            f.write(ablation_table + "\n")
        print("\nS_metadata field ablation (" + GRID_REFERENCE_ARM + f" pools, weights {abl['weights']}):\n" + ablation_table)

    with open(os.path.join(out_dir, "arms.json"), "w", encoding="utf-8") as f:
        json.dump({"config": args.config, "split": conf.get("enterprise_split"), "n_questions": len(data.all_queries),
                   "star_weights": star, "arms": summary}, f, indent=1, ensure_ascii=False)
    table = write_tables(out_dir, summary, data.question_types or [])
    print(f"\n{table}\n\nwritten to {out_dir}")


if __name__ == "__main__":
    main()
