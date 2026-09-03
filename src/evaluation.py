import hashlib
import json
import logging
import os
import numpy as np

from typing import Dict, List, Optional, Tuple, Callable
from bisect import insort_right
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from rouge_score import rouge_scorer

from src.dataset import DataManager
from src.prompt.rag_judge import get_judge_template, parse_judge_response
from src.utils import normalize_answer


class Evaluator:
    
    def __init__(self,
                 data: DataManager,
                 top_k_nodes_per_layer: int = 10,
                 judge_model=None,
                 judge_workers: int = 8,
                 judge_cache_path: Optional[str] = None) -> None:
        
        self.data: DataManager = data

        rc_k_list = [1, 2, 5, 10, 20, 50, 100, 200]
        if top_k_nodes_per_layer not in rc_k_list:
            insort_right(rc_k_list, top_k_nodes_per_layer)
        self.rc_k_list: List[int] = rc_k_list[:rc_k_list.index(top_k_nodes_per_layer) + 1]
        # Document-level recall cut-offs (EnterpriseRAG-Bench): k <= top_k plus the whole ranked list.
        self.doc_k_list: List = [k for k in (1, 2, 5, 10, 20) if k <= top_k_nodes_per_layer] + ["all"]

        self.judge_model = judge_model
        self.judge_workers: int = max(int(judge_workers or 1), 1)
        self.judge_cache_path: Optional[str] = judge_cache_path

    def qa_exactmatch(self, 
                      predicted_answers: List[str],
                      aggregation_fn: Callable = np.max
                      ) -> Dict[str, float]:
        """
        Calculates the Exact Match (EM) score.

        Args:
            gold_answers (List[List[str]]): List of lists containing ground truth answers.
            predicted_answers (List[str]): List of predicted answers.
            aggregation_fn (Callable): Function to aggregate scores across multiple gold answers (default: np.max).

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]:
                - A dictionary with the averaged EM score.
                - A list of dictionaries with EM scores for each example.
        """
        assert len(self.data.gold_answers) == len(
            predicted_answers), "Length of gold answers and predicted answers should be the same."

        example_eval_results = []
        total_em = 0

        for gold_list, predicted in zip(self.data.gold_answers, predicted_answers):
            em_scores = [1.0 if normalize_answer(gold) == normalize_answer(predicted) else 0.0 for gold in gold_list]
            aggregated_em = aggregation_fn(em_scores)
            example_eval_results.append({"ExactMatch": aggregated_em})
            total_em += aggregated_em

        avg_em = total_em / len(self.data.gold_answers) if self.data.gold_answers else 0.0
        pooled_eval_results = {"ExactMatch": avg_em}

        return pooled_eval_results

    def qa_f1(self,
              predicted_answers: List[str],
              aggregation_fn: Callable = np.max
              ) -> Dict[str, float]:
        """
        Calculates the F1 score.

        Args:
            gold_answers (List[List[str]]): List of lists containing ground truth answers.
            predicted_answers (List[str]): List of predicted answers.
            aggregation_fn (Callable): Function to aggregate scores across multiple gold answers (default: np.max).

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]:
                - A dictionary with the averaged F1 score.
                - A list of dictionaries with F1 scores for each example.
        """
        assert len(self.data.gold_answers) == len(
            predicted_answers), "Length of gold answers and predicted answers should be the same."

        def compute_f1(gold: str, predicted: str) -> float:
            gold_tokens = normalize_answer(gold).split()
            predicted_tokens = normalize_answer(predicted).split()
            common = Counter(predicted_tokens) & Counter(gold_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                return 0.0

            precision = 1.0 * num_same / len(predicted_tokens)
            recall = 1.0 * num_same / len(gold_tokens)
            return 2 * (precision * recall) / (precision + recall)

        example_eval_results = []
        total_f1 = 0.0

        for gold_list, predicted in zip(self.data.gold_answers, predicted_answers):
            f1_scores = [compute_f1(gold, predicted) for gold in gold_list]
            aggregated_f1 = aggregation_fn(f1_scores)
            example_eval_results.append({"F1": aggregated_f1})
            total_f1 += aggregated_f1

        avg_f1 = total_f1 / len(self.data.gold_answers) if self.data.gold_answers else 0.0
        pooled_eval_results = {"F1": avg_f1}

        return pooled_eval_results

    def qa_rouge(self, predicted_answers: List[str], aggregation_fn: Callable = np.max):

        assert len(self.data.gold_answers) == len(
            predicted_answers), "Length of gold answers and predicted answers should be the same."
        
        def get_rouge_type(name: str) -> Tuple[str, str]:
            name = name.split("-", maxsplit=2)
            
            return {
                "L": "rougeL",
                "1": "rouge1",
                "2": "rouge2",
            }[name[1]], {
                "P": "precision",
                "R": "recall",
                "F": "fmeasure",
            }[name[2]]

        example_eval_results = []
        total_rouge = {
            "ROUGE-L-P": 0.0,
            "ROUGE-L-R": 0.0,
            "ROUGE-L-F": 0.0,
            "ROUGE-1-P": 0.0,
            "ROUGE-1-R": 0.0,
            "ROUGE-1-F": 0.0,
            "ROUGE-2-P": 0.0,
            "ROUGE-2-R": 0.0,
            "ROUGE-2-F": 0.0,
        }
        
        for gold_list, predicted in zip(self.data.gold_answers, predicted_answers):
            scorer = rouge_scorer.RougeScorer(['rougeL', 'rouge1', 'rouge2'], use_stemmer=True)
            rouge_scores = [scorer.score(prediction=predicted, target=gold) for gold in gold_list]

            for metric in total_rouge.keys():
                rouge_type = get_rouge_type(metric)
                aggregated_rouge = aggregation_fn([getattr(s[rouge_type[0]], rouge_type[1]) for s in rouge_scores])
                example_eval_results.append({metric: aggregated_rouge})
                total_rouge[metric] += aggregated_rouge

        pooled_eval_results = {}
        for metric, value in total_rouge.items():
            pooled_eval_results[metric] = value / len(self.data.gold_answers) if self.data.gold_answers else 0.0

        return pooled_eval_results
        
    def qa_answer_rate(self, predicted_answers: List[str]):
        
        total_answer_rate = {
            "false_rate": 0.0,
            "not_mentioned_rate": 0.0,
            "error_rate": 0.0,
        }

        for gold_list, predicted in zip(self.data.gold_answers, predicted_answers):
            if predicted == "Error":
                total_answer_rate["error_rate"] += 1
                continue

            predicted_tokens = normalize_answer(predicted).split()
            if any([{"not", "mentioned"}.issubset(predicted_tokens), 
                    {"not", "specified"}.issubset(predicted_tokens),
                    {"not", "stated"}.issubset(predicted_tokens)]):
                total_answer_rate["not_mentioned_rate"] += 1
                continue

            gold_tokens = set()
            for gold in gold_list:    
                gold_tokens |= set(normalize_answer(gold).split())
            
            common = Counter(predicted_tokens) & Counter(gold_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                total_answer_rate["false_rate"] += 1

        pooled_eval_results = {}
        for metric, value in total_answer_rate.items():
            pooled_eval_results[metric] = value / len(self.data.gold_answers) if self.data.gold_answers else 0.0

        return pooled_eval_results

    def rt_recall(self, retrieved_docs: List[List[str]]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        
        self.rc_k_list = sorted(set(self.rc_k_list))
            
        example_eval_results = []
        pooled_eval_results = {f"Recall@{k}": 0.0 for k in self.rc_k_list}
        for example_gold_docs, example_retrieved_docs in zip(self.data.gold_docs, retrieved_docs):
            if len(example_retrieved_docs) < self.rc_k_list[-1]:
                tqdm.write(f"Warning: Length of retrieved docs ({len(example_retrieved_docs)}) is smaller than largest topk for recall score ({self.rc_k_list[-1]})")
            
            example_eval_result = {f"Recall@{k}": 0.0 for k in self.rc_k_list}

            # Compute Recall@k for each k
            for k in self.rc_k_list:
                # Get top-k retrieved documents
                top_k_docs = example_retrieved_docs[:k]
                # Calculate intersection with gold documents
                relevant_retrieved = set(top_k_docs) & set(example_gold_docs)
                # Compute recall
                if example_gold_docs:  # Avoid division by zero
                    example_eval_result[f"Recall@{k}"] = len(relevant_retrieved) / len(set(example_gold_docs))
                else:
                    example_eval_result[f"Recall@{k}"] = 0.0
            
            # Append example results
            example_eval_results.append(example_eval_result)
            
            # Accumulate pooled results
            for k in self.rc_k_list:
                pooled_eval_results[f"Recall@{k}"] += example_eval_result[f"Recall@{k}"]

        # Average pooled results over all examples
        for k in self.rc_k_list:
            pooled_eval_results[f"Recall@{k}"] /= len(self.data.gold_docs)

        # round off to 4 decimal places for pooled results
        pooled_eval_results = {k: round(v, 4) for k, v in pooled_eval_results.items()}
        return pooled_eval_results
        
    # ------------------------------------------------------------------ EnterpriseRAG-Bench metrics
    def _by_question_type(self, per_example: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Average per-example scores per question_type (skips examples whose score is None)."""
        types = getattr(self.data, "question_types", None)
        if not types:
            return {}
        out: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        for metric, values in per_example.items():
            for qtype, value in zip(types, values):
                if value is not None:
                    out[qtype][metric].append(value)
        return {qtype: {m: round(float(np.mean(v)), 4) for m, v in metrics.items() if v}
                for qtype, metrics in sorted(out.items())}

    def rt_doc_recall(self, retrieved_doc_ids: List[List[str]]) -> Dict:
        """
        Document-level Recall@k against `data.gold_doc_ids` (EnterpriseRAG-Bench expected_doc_ids).
        Questions without gold documents (info_not_found / high_level) are skipped and counted.
        """
        gold_lists = self.data.gold_doc_ids
        assert gold_lists is not None, "rt_doc_recall needs data.gold_doc_ids."
        assert len(gold_lists) == len(retrieved_doc_ids), \
            "Length of gold document ids and retrieved document ids should be the same."
        per_example: Dict[str, List[Optional[float]]] = {f"DocRecall@{k}": [] for k in self.doc_k_list}
        evaluated = 0
        for gold, retrieved in zip(gold_lists, retrieved_doc_ids):
            gold_set = set(gold or [])
            if not gold_set:
                for k in self.doc_k_list:
                    per_example[f"DocRecall@{k}"].append(None)
                continue
            evaluated += 1
            ranked = list(dict.fromkeys(retrieved or []))
            for k in self.doc_k_list:
                top = ranked if k == "all" else ranked[:k]
                per_example[f"DocRecall@{k}"].append(len(set(top) & gold_set) / len(gold_set))
        results = {
            metric: round(float(np.mean([v for v in values if v is not None])), 4) if evaluated else 0.0
            for metric, values in per_example.items()
        }
        results["DocRecall_n_evaluated"] = evaluated
        results["DocRecall_n_skipped"] = len(gold_lists) - evaluated
        results["DocRecall_by_type"] = self._by_question_type(per_example)
        return results

    def rt_invalid_extra_docs(self, retrieved_doc_ids: List[List[str]]) -> Dict:
        """Mean number of retrieved documents that are not gold (lower is better)."""
        gold_lists = self.data.gold_doc_ids
        assert gold_lists is not None, "rt_invalid_extra_docs needs data.gold_doc_ids."
        per_example: Dict[str, List[Optional[float]]] = {"InvalidExtraDocs": [], "InvalidExtraDocs@5": []}
        for gold, retrieved in zip(gold_lists, retrieved_doc_ids):
            gold_set = set(gold or [])
            if not gold_set:
                per_example["InvalidExtraDocs"].append(None)
                per_example["InvalidExtraDocs@5"].append(None)
                continue
            ranked = list(dict.fromkeys(retrieved or []))
            per_example["InvalidExtraDocs"].append(float(len(set(ranked) - gold_set)))
            per_example["InvalidExtraDocs@5"].append(float(len(set(ranked[:5]) - gold_set)))
        results = {}
        for metric, values in per_example.items():
            valid = [v for v in values if v is not None]
            results[metric] = round(float(np.mean(valid)), 4) if valid else 0.0
        results["InvalidExtraDocs_by_type"] = self._by_question_type(per_example)
        return results

    def _ranked_doc_metric(self, retrieved_doc_ids: List[List[str]], name: str, fn) -> Dict:
        """Shared loop for document-level metrics; fn(ranked, gold_set) -> {metric: value}."""
        gold_lists = self.data.gold_doc_ids
        assert gold_lists is not None, f"{name} needs data.gold_doc_ids."
        assert len(gold_lists) == len(retrieved_doc_ids), \
            "Length of gold document ids and retrieved document ids should be the same."
        per_example: Dict[str, List[Optional[float]]] = defaultdict(list)
        keys: List[str] = []
        for gold, retrieved in zip(gold_lists, retrieved_doc_ids):
            gold_set = set(gold or [])
            values = fn(list(dict.fromkeys(retrieved or [])), gold_set) if gold_set else None
            if values is not None and not keys:
                keys = list(values.keys())
            for key in (keys or (list(values.keys()) if values else [])):
                per_example[key].append(None if values is None else values[key])
        results = {}
        for metric, values in per_example.items():
            valid = [v for v in values if v is not None]
            results[metric] = round(float(np.mean(valid)), 4) if valid else 0.0
        results[f"{name}_by_type"] = self._by_question_type(dict(per_example))
        return results

    def rt_doc_mrr(self, retrieved_doc_ids: List[List[str]]) -> Dict:
        """Mean reciprocal rank of the first gold document (questions without gold docs skipped)."""
        def fn(ranked, gold_set):
            for rank, doc in enumerate(ranked, 1):
                if doc in gold_set:
                    return {"DocMRR": 1.0 / rank}
            return {"DocMRR": 0.0}
        return self._ranked_doc_metric(retrieved_doc_ids, "DocMRR", fn)

    def rt_doc_ndcg(self, retrieved_doc_ids: List[List[str]]) -> Dict:
        """nDCG@k with binary gains over gold documents, k in {5, 10} (capped by the ranked list)."""
        ks = [k for k in (5, 10) if k <= max(self.doc_k_list[:-1] or [10])] or [5]

        def fn(ranked, gold_set):
            out = {}
            for k in ks:
                dcg = sum(1.0 / np.log2(i + 2) for i, doc in enumerate(ranked[:k]) if doc in gold_set)
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold_set), k)))
                out[f"DocNDCG@{k}"] = dcg / idcg if idcg > 0 else 0.0
            return out
        return self._ranked_doc_metric(retrieved_doc_ids, "DocNDCG", fn)

    def rt_doc_recall_leaf_only(self, retrieved_doc_ids: List[List[str]]) -> Dict:
        """DocRecall computed on leaf-only document ids (abstract nodes ignored), keys suffixed "Leaf"."""
        results = self.rt_doc_recall(retrieved_doc_ids)
        out = {}
        for key, value in results.items():
            if key.startswith("DocRecall_"):
                continue
            out[key.replace("DocRecall@", "DocRecallLeaf@")] = value
        out["DocRecallLeaf_by_type"] = {
            qtype: {k.replace("DocRecall@", "DocRecallLeaf@"): v for k, v in metrics.items()}
            for qtype, metrics in results.get("DocRecall_by_type", {}).items()
        }
        return out

    def _load_judge_cache(self) -> Dict:
        if self.judge_cache_path and os.path.exists(self.judge_cache_path):
            try:
                with open(self.judge_cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def _save_judge_cache(self, cache: Dict) -> None:
        if not self.judge_cache_path:
            return
        os.makedirs(os.path.dirname(self.judge_cache_path) or ".", exist_ok=True)
        with open(self.judge_cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)

    def qa_llm_judge(self, predicted_answers: List[str]) -> Dict:
        """
        LLM judge (EnterpriseRAG-Bench style): correctness (binary) and completeness (share of
        answer_facts covered). Overall = mean(correct x completeness). Results are cached per
        (question_id, answer) so re-running the evaluation does not repeat judge calls.
        """
        assert self.judge_model is not None, "qa_llm_judge needs a judge model (conf['judge_name'])."
        assert len(self.data.data) == len(predicted_answers), \
            "Length of questions and predicted answers should be the same."
        cache = self._load_judge_cache()
        question_ids = getattr(self.data, "question_ids", None) or [str(i) for i in range(len(predicted_answers))]
        facts_list = getattr(self.data, "answer_facts", None) or [[] for _ in predicted_answers]
        question_types = getattr(self.data, "question_types", None) or [None] * len(predicted_answers)

        def judge_one(i: int) -> Dict:
            sample = self.data.data[i]
            answer = predicted_answers[i] if predicted_answers[i] is not None else ""
            key = f"{question_ids[i]}:{hashlib.sha1(answer.encode('utf-8')).hexdigest()[:12]}"
            if key in cache:
                return cache[key]
            messages = get_judge_template(
                sample.get("question", self.data.all_queries[i]),
                sample.get("gold_answer", ""),
                facts_list[i],
                answer,
                question_type=question_types[i],
            )
            response = self.judge_model.qa(question=messages, max_tokens=400)
            if not isinstance(response, str):
                response = ""
            result = parse_judge_response(response, len(facts_list[i]))
            result["raw"] = response[:2000]
            cache[key] = result
            return result

        with ThreadPoolExecutor(max_workers=self.judge_workers) as executor:
            verdicts = list(tqdm(executor.map(judge_one, range(len(predicted_answers))),
                                 total=len(predicted_answers), desc="llm judge"))
        self._save_judge_cache(cache)

        correct = [1.0 if v["correct"] else 0.0 for v in verdicts]
        completeness = [float(v["completeness"]) for v in verdicts]
        overall = [c * m for c, m in zip(correct, completeness)]
        per_example = {"JudgeOverall": overall, "JudgeCorrectness": correct, "JudgeCompleteness": completeness}
        return {
            "JudgeOverall": round(float(np.mean(overall)), 4) if overall else 0.0,
            "JudgeCorrectness": round(float(np.mean(correct)), 4) if correct else 0.0,
            "JudgeCompleteness": round(float(np.mean(completeness)), 4) if completeness else 0.0,
            "JudgeParseErrors": sum(1 for v in verdicts if v.get("error")),
            "Judge_by_type": self._by_question_type(per_example),
        }

    def evaluate(self, answers: List[str] = None, retrieved_docs: List[List[str]] = None, 
                 metrics: str | Tuple[str] = "all", retrieved_doc_ids: List[List[str]] = None,
                 retrieved_doc_ids_leaf_only: List[List[str]] = None) -> Dict:

        implemented_metrics = set(["em", "f1", "rouge", "rsim", "recall", "answerrate", "llmjudge",
                                   "docrecall", "extradocs", "docmrr", "docndcg"])
        if metrics == "all":
            metrics = implemented_metrics
        elif isinstance(metrics, (list, tuple, set)):
            metrics = set(metrics)
            assert metrics.issubset(implemented_metrics), "Some evaluation metrics are not supported."
        else:
            raise ValueError(f"Invalid evaluation metric(s) \"{metrics}\".")

        overall_results = {}

        def run(name: str, fn, **kwargs):
            try:
                overall_results.update(fn(**kwargs))
            except Exception as e:
                logging.exception(f"Evaluation metric '{name}' failed")
                print(f"[eval] metric '{name}' failed: {e!r}")
        
        if answers is not None and self.data.gold_answers is not None:
            if "em" in metrics:
                run("em", self.qa_exactmatch, predicted_answers=answers, aggregation_fn=np.max)
            if "f1" in metrics:
                run("f1", self.qa_f1, predicted_answers=answers, aggregation_fn=np.max)
            if "rouge" in metrics:
                run("rouge", self.qa_rouge, predicted_answers=answers, aggregation_fn=np.max)
            if "answerrate" in metrics:
                run("answerrate", self.qa_answer_rate, predicted_answers=answers)
            # round off to 4 decimal places for QA results
            overall_results = {k: round(float(v), 4) for k, v in overall_results.items()}
        if answers is not None and "llmjudge" in metrics and self.judge_model is not None:
            run("llmjudge", self.qa_llm_judge, predicted_answers=answers)
        if retrieved_docs is not None and self.data.gold_docs is not None:
            if "recall" in metrics:
                run("recall", self.rt_recall, retrieved_docs=retrieved_docs)
        if retrieved_doc_ids is not None and getattr(self.data, "gold_doc_ids", None) is not None:
            if "docrecall" in metrics:
                run("docrecall", self.rt_doc_recall, retrieved_doc_ids=retrieved_doc_ids)
            if "extradocs" in metrics:
                run("extradocs", self.rt_invalid_extra_docs, retrieved_doc_ids=retrieved_doc_ids)
            if "docmrr" in metrics:
                run("docmrr", self.rt_doc_mrr, retrieved_doc_ids=retrieved_doc_ids)
            if "docndcg" in metrics:
                run("docndcg", self.rt_doc_ndcg, retrieved_doc_ids=retrieved_doc_ids)
            if "docrecall" in metrics and retrieved_doc_ids_leaf_only is not None:
                run("docrecall_leaf", self.rt_doc_recall_leaf_only, retrieved_doc_ids=retrieved_doc_ids_leaf_only)
            
        return overall_results
