"""LLM prompt for query understanding (one JSON object with retrieval constraints)."""
import json
import re
from typing import Dict, List, Optional, Sequence

PROMPT_VERSION = "v2"

query_system = (
    "You extract retrieval constraints from a question asked over an enterprise knowledge base "
    "(Confluence pages, Google Drive documents, Jira / Linear tickets, GitHub pull requests, "
    "Slack channels, Gmail threads, Fireflies meeting transcripts, HubSpot deals).\n"
    "Return ONLY one JSON object with these keys:\n"
    "  \"intent\": one of \"lookup\" (a specific fact), \"aggregation\" (needs several documents or a "
    "summary), \"comparison\", \"timeline\", \"unknown\";\n"
    "  \"keywords\": 3-8 short search terms copied from the question (technical terms, product names);\n"
    "  \"entities\": customer / company / product / system names mentioned;\n"
    "  \"people\": person names mentioned (empty if none);\n"
    "  \"projects\": project, initiative or codename names mentioned;\n"
    "  \"ticket_keys\": identifiers like INC-1234, ADR-007, PROJ-42;\n"
    "  \"time_range\": {\"start\": \"YYYY-MM-DD\", \"end\": \"YYYY-MM-DD\"} for an explicit or implied "
    "date window (a single date -> start == end; \"Q4 2025\" -> 2025-10-01..2025-12-31; "
    "\"early March 2026\" -> 2026-03-01..2026-03-10), or null when the question gives no date;\n"
    "  \"source_types\": ONLY systems the question names explicitly (\"in Slack\", \"the Jira ticket\", "
    "\"a Confluence page\", \"the email thread\"), as a list drawn from {source_types}. Never infer a system "
    "from content words such as ticket, incident, channel, meeting, call, note, pull request, deal, runbook, "
    "playbook, listing, doc: those live in any system here. Empty when no system is named;\n"
    "  \"channels\": Slack channel names mentioned (without #).\n"
    "Do not guess: leave lists empty and time_range null when the question does not say. "
    "Never invent dates from relative words like \"recent\" or \"last quarter\"."
)


def get_query_template(question: str, source_types: Sequence[str], project_hint: Optional[Sequence[str]] = None
                       ) -> List[Dict[str, str]]:
    system = query_system.replace("{source_types}", ", ".join(source_types))
    hint = ""
    if project_hint:
        hint = "Known project / product names (use these spellings when they occur): " + \
               ", ".join(list(project_hint)[:60]) + "\n\n"
    user = f"{hint}Question: {question}\n\nJSON:"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_query_response(text) -> Optional[Dict]:
    """First JSON object in the model output (tolerates fences / prose); None when unusable."""
    if not isinstance(text, str) or not text.strip():
        return None
    body = text.strip()
    m = re.search(r"\{.*\}", body, re.DOTALL)
    if not m:
        return None
    raw = m.group(0)
    for candidate in (raw, raw.replace("\n", " ").replace("True", "true").replace("False", "false").replace("None", "null")):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return data if isinstance(data, dict) else None
    return None
