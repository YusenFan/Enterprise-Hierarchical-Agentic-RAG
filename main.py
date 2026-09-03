import os
import time
import argparse

from typing import List
from tqdm import tqdm
from threading import Lock
from huggingface_hub import hf_hub_download
from concurrent.futures import ThreadPoolExecutor, as_completed

from conf import apply_config_overrides, read_config
from src import (
    RAG,
    DataManager,
    Evaluator,
    HopDiscriminator,
    OpenAIAbstractModel,
    OpenAIEmbeddingModel,
    OpenAIQAModel,
    OllamaAbstractModel,
    OllamaEmbeddingModel,
    OllamaQAModel,
    VLLMEmbeddingModel,
    VLLMAbstractModel,
    VLLMQAModel,
    VLLMRerankModel,
    SentenceTransformersEmbeddingModel,
    TransformersAbstractModel,
    TransformersEmbeddingModel,
    TransformersQAModel,
    TransformersRerankModel,
)
from src.dataset import enterprise_kwargs_from_conf, split_dataset
from src.metadata import collect_sources
from src.query.credit import document_credit, merge_node_scores
from src.model.factory import build_model
from src.prompt import AgentPrompt, get_qa_template
from src.utils import (
    get_token_length,
    load_answers,
    parse_response,
    save_answers,
    result_stem,
    is_bucketed_tree,
    get_tree_save_name,
    remove_tree_target,
    print_tree_check,
)


def main():
    if conf.get("read_local_pdf") is not None:
        raise ValueError("The main evaluation script does not support pdf mode. "
                         "Build your custom index with index.py. ")

    data = DataManager(
        dataset_name=conf["dataset"],
        data_dir=conf["data_dir"],
        test_samples=conf["test_samples"],
        enterprise_kwargs=enterprise_kwargs_from_conf(conf),
    )
    if getattr(data, "subset_stats", None):
        tqdm.write(f"EnterpriseRAG-Bench subset: {data.subset_stats}")

    if conf.get("tree_build_diagnostics"):
        if isinstance(data.all_passages, str):
            passages = [data.all_passages]
        else:
            passages = data.get_documents()
        tqdm.write(f"Dataset token stats: {get_token_length(passages)}")

    top_k = (
        min(conf["tree_top_k"], conf["rerank_top_k"])
        if conf["rerank_top_k"] is not None
        else conf["tree_top_k"]
    )
    hybrid_mode = conf.get("retrieve_mode") == "hybrid_score"
    if hybrid_mode and conf["rerank_top_k"] is not None:
        top_k = conf["rerank_top_k"]   # hybrid_score returns rerank_top_k nodes of any layer
    judge_cache_path = (
        os.path.join(conf["save_dir"], "results", f"{result_stem(conf)}_judge.json")
        if conf["save_dir"] is not None else None
    )
    evaluator = Evaluator(
        data=data,
        top_k_nodes_per_layer=top_k,
        judge_workers=conf.get("judge_workers", 8),
        judge_cache_path=judge_cache_path,
    )

    if not os.path.exists(conf["log_path"]):
        os.makedirs(conf["log_path"])
    logger = open(os.path.join(conf["log_path"], f"{result_stem(conf)}.log"), "a")

    results = None
    if conf["save_dir"] is not None:
        results = load_answers(conf)

    if not results:
        def set_model(model_name, task_type):
            return build_model(model_name, task_type, conf)

        model_pool = ("qa",) if conf["no_retrieval"] else tuple(
            model_type.rsplit("_name", maxsplit=1)[0]
            for model_type in conf.keys()
            if model_type.endswith("_name") and conf[model_type] is not None
        )
        models_to_prepare = [model_type for model_type in conf.keys()
            if model_type.endswith("_name") and conf[model_type] is not None]
        models_to_prepare = [
            model_type for model_type in models_to_prepare
            if model_type.rsplit("_name", maxsplit=1)[0] in model_pool
        ]
        for model_type in models_to_prepare:
            task_type = model_type.rsplit("_name", maxsplit=1)[0]
            conf[f"{task_type}_model"] = set_model(conf[model_type], task_type)

        # Split after the models are ready: semantic chunking embeds every sentence.
        split_dataset(data, conf)

        tree_rag = None
        if not conf["no_retrieval"]:
            tree_rag = RAG(conf)

            if conf["save_dir"] is not None:
                os.makedirs(conf["save_dir"], exist_ok=True)
                save_tree_name = get_tree_save_name(conf)
                save_tree_path = os.path.join(conf["save_dir"], save_tree_name)
                tree_target_name = "directory" if is_bucketed_tree(conf) else "file"

                if conf["force_index_from_scratch"]:
                    tqdm.write("Constructing tree index...")
                    if os.path.exists(save_tree_path):
                        remove_tree_target(save_tree_path)
                    tree_rag.add_documents(data)
                    if conf.get("tree_build_diagnostics"):
                        print_tree_check(tree_rag.tree)
                    tree_rag.save("tree", save_tree_path)
                    tqdm.write(
                        f'Tree construction completed! Tree {tree_target_name} saved to "{save_tree_path}".'
                    )
                elif not os.path.exists(save_tree_path):
                    tqdm.write("Downloading tree index file from Hugging Face...")
                    try:
                        REPO_ID = "Newiz430/Psi-RAG"
                        hf_hub_download(
                            repo_id=REPO_ID,
                            filename=save_tree_name,
                            local_dir=conf["save_dir"],
                        )
                        tree_rag.load("tree", save_tree_path)
                        tqdm.write(
                            f'Downloading index completed! Tree {tree_target_name} saved to "{save_tree_path}".'
                        )
                    except Exception as e:
                        tqdm.write(
                            f"Downloading index failed as an exception occured: \n{e}\nConstructing tree index from scratch..."
                        )
                        tree_rag.add_documents(data)
                        if conf.get("tree_build_diagnostics"):
                            print_tree_check(tree_rag.tree)
                        tree_rag.save("tree", save_tree_path)
                        tqdm.write(
                            f'Tree construction completed! Tree {tree_target_name} saved to "{save_tree_path}".'
                        )
                else:
                    tqdm.write(
                        f'Loading tree from existing index {tree_target_name} "{save_tree_path}"...'
                    )
                    tree_rag.load("tree", save_tree_path)
                    tqdm.write("Loading tree completed!")
            else:
                tqdm.write("Constructing tree index...")
                tree_rag.add_documents(data)
                if conf.get("tree_build_diagnostics"):
                    print_tree_check(tree_rag.tree)
                tqdm.write("Tree construction completed!")

            if conf["hybrid_search"]:
                tqdm.write("Constructing sparse index (BM25)...")
                tree_rag.build_vocab(data)
                tqdm.write("Sparse index construction completed!")

            if conf["max_retrieval_time"] == "auto":
                conf["hop_discriminator"] = HopDiscriminator(conf).prepare()

        all_qa_time = []
        all_answers = {}
        all_contexts = {}
        all_sources = {}
        all_diag = {}

        def sources_for(top_k_scores, query_id):
            tree = tree_rag.tree[data.query_to_doc_ids[query_id]] if isinstance(tree_rag.tree, List) else tree_rag.tree
            return document_credit(tree, top_k_scores, conf.get("source_max_abstract_docs"))

        def get_max_retrieval_time_verbose_lines(query_id, predicted_hop_label):
            if conf["max_retrieval_time"] != "auto" or not conf["verbose"]:
                return []
            try:
                actual_hop_count = HopDiscriminator._extract_hop_count(
                    conf["dataset"], data.data[query_id]
                )
                actual_hop_label = HopDiscriminator._to_hop_label(actual_hop_count)
            except NotImplementedError:
                actual_hop_label = "NA"
            return [
                f"real max retrieval time: {actual_hop_label}",
                f"predicted max retrieval time: {predicted_hop_label}",
            ]

        def qa(query, query_id, tokenizer_lock=None):
            start_time = time.time()
            query_embedding = None
            max_retrieval_time_verbose_lines = []

            if conf["no_retrieval"]:
                qa_message = get_qa_template(
                    query,
                    "",
                    type=conf["answer_type"],
                    add_abstract_to_context=conf["abstract_layer_as_context"] > 0,
                    thought_max_length=conf["max_response_length"],
                )
                raw_answer = conf["qa_model"].qa(
                    question=qa_message,
                    max_tokens=2 * conf["max_response_length"],
                )

                try:
                    thought, answer = raw_answer.rsplit("Answer:", maxsplit=1)
                    answer = answer.strip(" .")
                except (IndexError, ValueError):
                    thought = ""
                    answer = raw_answer

                qa_time = time.time() - start_time
                context = [""] * top_k
                output = "\n".join(
                    [
                        f"\nid: {query_id}",
                        f"question: {query}",
                        f"thoughts: {thought}",
                        f"answer: {answer}",
                        f"gold answer: {data.gold_answers[query_id] if data.gold_answers is not None else 'NA'}",
                        "\n",
                    ]
                )

                if conf["verbose"]:
                    tqdm.write(output)
                logger.write(output)

                return query_id, answer, context, qa_time, [], {}

            if conf["max_retrieval_time"] == "auto":
                query_embedding = conf["embed_model"].embed(query)
                predicted_hop_label = conf["hop_discriminator"].predict_from_embedding(
                    query_embedding
                )
                max_retrieval_time = max(predicted_hop_label - 1, 0)
                max_retrieval_time_verbose_lines = get_max_retrieval_time_verbose_lines(
                    query_id,
                    predicted_hop_label,
                )
            elif isinstance(conf["max_retrieval_time"], int):
                max_retrieval_time = conf["max_retrieval_time"]
            else:
                raise ValueError(
                    'max_retrieval_time must be an integer or "auto".'
                )

            question_type = data.question_types[query_id] if getattr(data, "question_types", None) else None
            retrieval_extras = [{}]
            documents, layer_information = tree_rag.retrieve(
                query,
                data.query_to_doc_ids[query_id],
                tokenizer_lock=tokenizer_lock,
                query_embedding=query_embedding,
                extras=retrieval_extras[0],
                question_type=question_type,
            )

            def get_agentic_answer(query, documents, state_log, response=None):
                documents_text = "\n".join(documents)
                message = agent_prompt.get_template(query, documents_text, answer=response)
                response = tree_rag.qa(
                    question=message,
                    max_tokens=2 * conf["max_response_length"],
                )
                thought, action, info = parse_response(response, verbose=conf["verbose"])
                state_log["thought"].append(thought)
                if action == "answer":
                    return info
                if action == "retrieve":
                    state_log["subquestion"].append(info)
                    retrieval_extras.append({})
                    documents, layer_information = tree_rag.retrieve(
                        info,
                        data.query_to_doc_ids[query_id],
                        tokenizer_lock=tokenizer_lock,
                        extras=retrieval_extras[-1],
                        question_type=question_type,
                    )
                    state_log["retrieved_nodes"].append(layer_information)
                    return get_agentic_answer(query, documents, state_log, response)

                tqdm.write(f"Unexpected agent action: {action}")
                return "Error"

            if max_retrieval_time > 0:
                state_log = {
                    "subquestion": [query],
                    "retrieved_nodes": [layer_information],
                    "thought": [],
                }
                agent_prompt = AgentPrompt(
                    ans_type=conf["answer_type"],
                    max_retrieval_time=max_retrieval_time,
                    thought_max_length=conf["max_response_length"],
                )
                answer = get_agentic_answer(query, documents, state_log)
                qa_time = time.time() - start_time

                top_k_scores = merge_node_scores(state_log["retrieved_nodes"], top_k, all_layers=hybrid_mode)
                if isinstance(tree_rag.tree, List):
                    context = [
                        tree_rag.tree[data.query_to_doc_ids[query_id]].all_nodes[top_k_node_index].text
                        for top_k_node_index in top_k_scores.keys()
                    ]
                else:
                    context = [
                        tree_rag.tree.all_nodes[top_k_node_index].text
                        for top_k_node_index in top_k_scores.keys()
                    ]

                sources, sources_leaf = sources_for(top_k_scores, query_id)
                subquestions_text = "\n\t".join(state_log["subquestion"][1:])
                thoughts_text = "\n\t".join(state_log["thought"])
                output = "\n".join(
                    [
                        f"\nid: {query_id}",
                        f"question: {query}",
                        *max_retrieval_time_verbose_lines,
                        f"sub-questions: {subquestions_text}",
                        f"thoughts: {thoughts_text}",
                        f"answer: {answer}",
                        f"sources: {', '.join(s['document_id'] for s in sources) if sources else 'NA'}",
                        f"gold answer: {data.gold_answers[query_id] if data.gold_answers is not None else 'NA'}",
                        "\n",
                    ]
                )
            else:
                documents_text = "\n".join(documents)
                qa_message = get_qa_template(
                    query,
                    documents_text,
                    type=conf["answer_type"],
                    add_abstract_to_context=conf["abstract_layer_as_context"] > 0,
                    thought_max_length=conf["max_response_length"],
                )
                raw_answer = tree_rag.qa(
                    question=qa_message,
                    max_tokens=2 * conf["max_response_length"],
                )

                try:
                    thought, answer = raw_answer.rsplit("Answer:", maxsplit=1)
                    answer = answer.strip(" .")
                except (IndexError, ValueError):
                    thought = ""
                    answer = raw_answer

                qa_time = time.time() - start_time

                top_k_scores = merge_node_scores([layer_information], top_k, all_layers=hybrid_mode)
                if isinstance(tree_rag.tree, List):
                    context = [
                        tree_rag.tree[data.query_to_doc_ids[query_id]].all_nodes[top_k_node_index].text
                        for top_k_node_index in top_k_scores.keys()
                    ]
                else:
                    context = [
                        tree_rag.tree.all_nodes[top_k_node_index].text
                        for top_k_node_index in top_k_scores.keys()
                    ]
                while len(context) < top_k:
                    context.append("")
                sources, sources_leaf = sources_for(top_k_scores, query_id)

                output = "\n".join(
                    [
                        f"\nid: {query_id}",
                        f"question: {query}",
                        *max_retrieval_time_verbose_lines,
                        f"thoughts: {thought}",
                        f"answer: {answer}",
                        f"sources: {', '.join(s['document_id'] for s in sources) if sources else 'NA'}",
                        f"gold answer: {data.gold_answers[query_id] if data.gold_answers is not None else 'NA'}",
                        "\n",
                    ]
                )

            if conf["verbose"]:
                tqdm.write(output)
            logger.write(output)

            diag = {
                "sources_leaf_only": sources_leaf,
                "query_parses": [e.get("query_parse") for e in retrieval_extras if e],
                "retrieval": retrieval_extras if conf.get("save_retrieval_diagnostics") else None,
            }
            return query_id, answer, context, qa_time, sources, diag

        tqdm.write("Answering questions...")
        if conf["multithreading_qa_batch_size"] > 1:
            tokenizer_lock = Lock()
            if hasattr(conf["embed_model"], "load_model"):
                conf["embed_model"].load_model()
            if (
                conf["rerank"]
                and conf["rerank_model"] is not None
                and hasattr(conf["rerank_model"], "load_model")
            ):
                conf["rerank_model"].load_model()

            bar = tqdm(
                range(0, len(data.all_queries), conf["multithreading_qa_batch_size"]),
                desc="qa",
            )
            for i in bar:
                with ThreadPoolExecutor() as executor:
                    future_qa_results = [
                        executor.submit(qa, query, query_id, tokenizer_lock)
                        for query_id, query in enumerate(
                            data.all_queries[i: i + conf["multithreading_qa_batch_size"]],
                            i,
                        )
                    ]
                    for future in as_completed(future_qa_results):
                        query_id, answer, context, qa_time, sources, diag = future.result()
                        all_answers[query_id] = answer
                        all_diag[query_id] = diag
                        all_contexts[query_id] = context
                        all_sources[query_id] = sources
                        all_qa_time.append(qa_time)
        else:
            bar = tqdm(data.all_queries, desc="qa")
            for query_id, query in enumerate(bar):
                _, answer, context, qa_time, sources, diag = qa(query, query_id)
                all_answers[query_id] = answer
                all_diag[query_id] = diag
                all_contexts[query_id] = context
                all_sources[query_id] = sources
                all_qa_time.append(qa_time)

        bar.close()

        results = {
            "answers": [ans[1] for ans in sorted(all_answers.items())],
            "retrieved_docs": [cont[1] for cont in sorted(all_contexts.items())],
            "sources": [src[1] for src in sorted(all_sources.items())],
            "retrieved_doc_ids": [[s["document_id"] for s in src[1]] for src in sorted(all_sources.items())],
            "retrieved_doc_ids_leaf_only": [
                [s["document_id"] for s in all_diag[q].get("sources_leaf_only", [])] for q in sorted(all_diag)
            ],
            "query_parses": [all_diag[q].get("query_parses") for q in sorted(all_diag)],
            "time": {
                "tb_time": tree_rag.tb_time if tree_rag is not None else -1,
                "tr_time": tree_rag.tr_time if tree_rag is not None else -1,
                "qa_time": sum(all_qa_time) / len(all_qa_time),
            },
        }
        if conf.get("save_retrieval_diagnostics"):
            results["retrieval_diagnostics"] = [all_diag[q].get("retrieval") for q in sorted(all_diag)]
        if conf["save_dir"] is not None:
            ans_path = os.path.join(conf["save_dir"], "results")
            save_answers(conf, results, ans_path)

        if tree_rag is None or tree_rag.retrieve_count == 0:
            retrieval_stats = (
                "Total times of retrieval: 0\n"
                "Average tree retrieval time: NA\n"
                "Average sparse retrieval time: NA\n"
                "Average rerank time: NA\n"
            )
        else:
            retrieval_stats = (
                f"Total times of retrieval: {tree_rag.retrieve_count}\n"
                f"Average tree retrieval time: {tree_rag.time_dict['tree'] / tree_rag.retrieve_count:.4f}s\n"
                f"Average sparse retrieval time: {tree_rag.time_dict['sparse'] / tree_rag.retrieve_count:.4f}s\n"
                f"Average rerank time: {tree_rag.time_dict['rerank'] / tree_rag.retrieve_count:.4f}s\n"
            )
        print(retrieval_stats)
        logger.writelines(retrieval_stats.splitlines())

    tqdm.write("Evaluating results...")
    metrics = conf["evaluation_metrics"]
    if (metrics == "all" or "llmjudge" in metrics) and conf.get("judge_name"):
        evaluator.judge_model = conf.get("judge_model") or build_model(conf["judge_name"], "judge", conf)
    scores = evaluator.evaluate(
        answers=results.get("answers", None),
        retrieved_docs=results.get("retrieved_docs", None),
        retrieved_doc_ids=results.get("retrieved_doc_ids", None),
        retrieved_doc_ids_leaf_only=results.get("retrieved_doc_ids_leaf_only", None),
        metrics=metrics,
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

    logger.writelines(final_eval_output.splitlines())
    logger.close()


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
