import argparse

from tqdm import tqdm

import os

from conf import apply_config_overrides, read_config
from src import DataManager, Evaluator
from src.dataset import enterprise_kwargs_from_conf, split_dataset
from src.model.factory import build_model
from src.utils import get_token_length, load_answers, result_stem


def build_judge(conf):
    """Judge model for the "llmjudge" metric, or None when not configured / not requested."""
    metrics = conf["evaluation_metrics"]
    wanted = metrics == "all" or "llmjudge" in metrics
    if not wanted or not conf.get("judge_name"):
        return None
    return build_model(conf["judge_name"], "judge", conf)


def main():
    if conf.get("read_local_pdf") is not None:
        raise ValueError("The evaluation script does not support custom pdf mode. ")
    
    data = DataManager(
        dataset_name=conf["dataset"],
        data_dir=conf["data_dir"],
        test_samples=conf["test_samples"],
        enterprise_kwargs=enterprise_kwargs_from_conf(conf),
    )

    if conf.get("tree_build_diagnostics"):
        if isinstance(data.all_passages, str):
            passages = [data.all_passages]
        else:
            passages = data.get_documents()
        tqdm.write(f"Dataset token stats: {get_token_length(passages)}")

    # No embedding model here, so a semantic config falls back to static splitting;
    # the Evaluator only reads gold answers / gold docs, never the chunks.
    split_dataset(data, conf)

    top_k = (
        min(conf["tree_top_k"], conf["rerank_top_k"])
        if conf["rerank_top_k"] is not None
        else conf["tree_top_k"]
    )
    judge_cache_path = (
        os.path.join(conf["save_dir"], "results", f"{result_stem(conf)}_judge.json")
        if conf["save_dir"] is not None else None
    )
    evaluator = Evaluator(
        data=data,
        top_k_nodes_per_layer=top_k,
        judge_model=build_judge(conf),
        judge_workers=conf.get("judge_workers", 8),
        judge_cache_path=judge_cache_path,
    )

    results = None
    if conf["save_dir"] is not None:
        conf["force_qa_from_scratch"] = False
        results = load_answers(conf)
    if not results:
        tqdm.write(
            f'Answer file not found. Please run "python qa.py --config {conf["config"]}" first.'
        )
        return

    tqdm.write("Evaluating results...")
    scores = evaluator.evaluate(
        answers=results.get("answers", None),
        retrieved_docs=results.get("retrieved_docs", None),
        retrieved_doc_ids=results.get("retrieved_doc_ids", None),
        retrieved_doc_ids_leaf_only=results.get("retrieved_doc_ids_leaf_only", None),
        metrics=conf["evaluation_metrics"],
    )

    if "time" in results:
        tree_building_time = (
            "NA" if results["time"]["tb_time"] < 0 else f'{results["time"]["tb_time"]:.2f}s'
        )
        single_retrieval_time = (
            "NA" if results["time"]["tr_time"] < 0 else f'{results["time"]["tr_time"]:.2f}s'
        )
        average_qa_time = (
            "NA" if results["time"]["qa_time"] < 0 else f'{results["time"]["qa_time"]:.2f}s'
        )
        final_eval_output = (
            f"Evaluation results: {scores}\n"
            f"Tree building time: {tree_building_time}\n"
            f"Single retrieval time: {single_retrieval_time}\n"
            f"Average QA time: {average_qa_time}\n"
        )
    else:
        final_eval_output = f"Evaluation results: {scores}\n"
    print(final_eval_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        metavar="CONFIG FILE NAME",
        help='Config file name (without ".py") under the "conf" directory.',
    )
    args, unknown_args = parser.parse_known_args()

    try:
        conf = read_config(conf_name=args.config)
        conf = apply_config_overrides(conf, unknown_args)
    except (FileNotFoundError, AttributeError, ValueError) as e:
        parser.error(str(e))

    main()
