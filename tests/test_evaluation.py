from types import SimpleNamespace

from src.evaluation import Evaluator
from src.prompt.rag_judge import get_judge_template, parse_judge_response


def make_evaluator(top_k=10):
    data = SimpleNamespace(
        gold_doc_ids=[["a", "b"], ["c"], [], ["d"]],
        question_types=["completeness", "basic", "info_not_found", "basic"],
        question_ids=["q1", "q2", "q3", "q4"],
        answer_facts=[["f1", "f2"], ["f1"], [], ["f1"]],
        gold_answers=None, gold_docs=None,
        data=[{"question": "q1", "gold_answer": "g1"}, {"question": "q2", "gold_answer": "g2"},
              {"question": "q3", "gold_answer": "not found"}, {"question": "q4", "gold_answer": "g4"}],
        all_queries=["q1", "q2", "q3", "q4"],
    )
    return Evaluator(data=data, top_k_nodes_per_layer=top_k)


def test_doc_recall_and_extra_docs():
    ev = make_evaluator()
    retrieved = [["x", "a", "a", "y"], ["c"], ["z"], ["p", "q"]]
    r = ev.rt_doc_recall(retrieved)
    assert r["DocRecall@1"] == round((0 + 1 + 0) / 3, 4)
    assert r["DocRecall@2"] == round((0.5 + 1 + 0) / 3, 4)
    assert r["DocRecall@all"] == round((0.5 + 1 + 0) / 3, 4)
    assert r["DocRecall_n_evaluated"] == 3 and r["DocRecall_n_skipped"] == 1
    assert r["DocRecall_by_type"]["basic"]["DocRecall@1"] == 0.5
    assert "info_not_found" not in r["DocRecall_by_type"]
    e = ev.rt_invalid_extra_docs(retrieved)
    assert e["InvalidExtraDocs"] == round((2 + 0 + 2) / 3, 4)
    assert e["InvalidExtraDocs_by_type"]["completeness"]["InvalidExtraDocs"] == 2.0


def test_evaluate_routes_doc_metrics():
    ev = make_evaluator()
    out = ev.evaluate(retrieved_doc_ids=[["a"], ["c"], [], ["d"]], metrics=["docrecall", "extradocs"])
    assert out["DocRecall@1"] == round((0.5 + 1 + 1) / 3, 4)
    assert out["InvalidExtraDocs"] == 0.0


def test_parse_judge_response_variants():
    assert parse_judge_response('{"correct": true, "facts_covered": [1, 2], "reason": "ok"}', 2) == {
        "correct": True, "completeness": 1.0, "facts_covered": [1, 2], "reason": "ok", "error": None}
    fenced = "```json\n{\"correct\": false, \"facts_covered\": [2, 7], \"reason\": \"partial\"}\n```"
    r = parse_judge_response(fenced, 4)
    assert r["correct"] is False and r["completeness"] == 0.25 and r["facts_covered"] == [2]
    prose = 'Sure. {"correct": "yes", "facts_covered": []} Done.'
    r = parse_judge_response(prose, 0)
    assert r["correct"] is True and r["completeness"] == 1.0
    r = parse_judge_response("no json here", 3)
    assert r["correct"] is False and r["completeness"] == 0.0 and r["error"]
    r = parse_judge_response("", 3)
    assert r["error"] == "empty response"


def test_judge_template_mentions_facts():
    messages = get_judge_template("Who?", "Alice", ["Alice owns it", "since 2025"], "Alice does", "basic")
    assert messages[0]["role"] == "system" and "JSON" in messages[0]["content"]
    assert "1. Alice owns it" in messages[1]["content"] and "Candidate answer: Alice does" in messages[1]["content"]


class FakeJudge:
    def qa(self, question, max_tokens=400, **kwargs):
        candidate = question[-1]["content"].split("Candidate answer:")[1].split("\n")[0].strip()
        if candidate == "right":
            return '{"correct": true, "facts_covered": [1], "reason": "ok"}'
        return '{"correct": false, "facts_covered": [], "reason": "no"}'


def test_llm_judge_with_cache(tmp_path):
    ev = make_evaluator()
    ev.judge_model = FakeJudge()
    ev.judge_cache_path = str(tmp_path / "judge.json")
    out = ev.qa_llm_judge(["right", "wrong", "right", "right"])
    # q1: correct, 1/2 facts -> 0.5 ; q2: 0 ; q3: correct, no facts -> 1.0 ; q4: 1/1 -> 1.0
    assert out["JudgeCorrectness"] == 0.75
    assert out["JudgeCompleteness"] == round((0.5 + 0 + 1 + 1) / 4, 4)
    assert out["JudgeOverall"] == round((0.5 + 0 + 1 + 1) / 4, 4)
    assert out["JudgeParseErrors"] == 0
    assert out["Judge_by_type"]["basic"]["JudgeOverall"] == 0.5
    # second run served from cache even with a judge that would now fail
    ev.judge_model = None
    ev.judge_model = type("Broken", (), {"qa": lambda self, question, **k: "garbage"})()
    assert ev.qa_llm_judge(["right", "wrong", "right", "right"])["JudgeOverall"] == out["JudgeOverall"]
