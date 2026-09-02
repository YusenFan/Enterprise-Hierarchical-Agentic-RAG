"""LLM-judge prompt for EnterpriseRAG-Bench (answer correctness x completeness)."""
import json
import re
from typing import Dict, List, Optional

judge_system = (
    "You are a strict grader for an enterprise question-answering benchmark. You receive a question, "
    "a reference answer, a numbered list of reference facts and a candidate answer.\n"
    "Grade ONLY against the reference; do not use outside knowledge.\n"
    "Rules:\n"
    "(1) \"correct\" is true only if the candidate's main claim agrees with the reference answer and "
    "contains no statement that contradicts the reference. Paraphrases, different formatting and extra "
    "correct detail are fine. Vague or hedged answers that do not commit to the reference's answer are incorrect.\n"
    "(2) If the reference says the information is not available / not found, the candidate is correct only "
    "if it also says so (e.g. \"Not mentioned\") and does not invent an answer.\n"
    "(3) \"facts_covered\" lists the indices (1-based) of reference facts that the candidate states or "
    "clearly implies. Leave it empty when no fact is covered.\n"
    "Respond with a single JSON object and nothing else: "
    "{\"correct\": true|false, \"facts_covered\": [int, ...], \"reason\": \"<one sentence>\"}"
)


def get_judge_template(question: str, gold_answer: str, answer_facts: List[str], candidate: str,
                       question_type: Optional[str] = None) -> List[Dict[str, str]]:
    facts = "\n".join(f"{i}. {fact}" for i, fact in enumerate(answer_facts or [], 1)) or "(none)"
    user = (
        f"Question: {question}\n"
        + (f"Question type: {question_type}\n" if question_type else "")
        + f"\nReference answer: {gold_answer}\n"
        f"\nReference facts:\n{facts}\n"
        f"\nCandidate answer: {candidate if candidate and candidate.strip() else '(empty)'}\n"
        "\nJSON:"
    )
    return [{"role": "system", "content": judge_system}, {"role": "user", "content": user}]


def parse_judge_response(text: str, n_facts: int) -> Dict:
    """Extract {"correct": bool, "completeness": float} from the judge output (tolerates fences / prose)."""
    result = {"correct": False, "completeness": 0.0, "facts_covered": [], "reason": None, "error": None}
    if not isinstance(text, str) or not text.strip():
        result["error"] = "empty response"
        return result
    body = text.strip()
    m = re.search(r"\{.*\}", body, re.DOTALL)
    if not m:
        result["error"] = "no JSON object found"
        return result
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        try:
            data = json.loads(m.group(0).replace("\n", " ").replace("True", "true").replace("False", "false"))
        except json.JSONDecodeError as e:
            result["error"] = f"invalid JSON: {e}"
            return result
    correct = data.get("correct")
    if isinstance(correct, str):
        correct = correct.strip().lower() in ("true", "yes", "1")
    result["correct"] = bool(correct)
    covered = data.get("facts_covered") or []
    if not isinstance(covered, list):
        covered = []
    valid = {int(i) for i in covered if isinstance(i, (int, float, str)) and str(i).strip().lstrip("-").isdigit()}
    valid = {i for i in valid if 1 <= i <= n_facts}
    result["facts_covered"] = sorted(valid)
    if n_facts > 0:
        result["completeness"] = len(valid) / n_facts
    else:
        result["completeness"] = 1.0 if result["correct"] else 0.0
    result["reason"] = data.get("reason")
    return result
