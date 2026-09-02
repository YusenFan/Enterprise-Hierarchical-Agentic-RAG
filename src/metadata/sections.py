"""Section splitting helpers for the pseudo-YAML and markdown-ish document formats."""
import re
from collections import OrderedDict
from typing import Dict, List, Tuple

from .patterns import display_name, looks_like_full_name

# "description:" / "review_comments:" at column 0, optionally followed by text on the same line.
PSEUDO_YAML_KEY_RE = re.compile(r"^([a-z][a-z0-9_]{1,40}):(?:[ \t]*$|[ \t]+(?=\S))")

KEY_SYNONYMS = {
    "summmary": "summary",
    "transcription": "transcript",
    "meeting_transcript": "transcript",
    "full_transcript_body": "transcript",
    "resolution_notes": "resolution",
    "resolution_summary": "resolution",
    "recent_activities": "timeline",
    "recent_activity": "timeline",
    "activity_timeline": "timeline",
    "review_conversation": "review_comments",
    "review_thread": "review_comments",
    "reviews": "review_comments",
    "conversation": "comments",
    "message_thread": "comments",
    "next_step": "next_steps",
    "action_item": "action_items",
    "notes_for_ops": "notes",
    "notes_for_engineering": "notes",
    "engineering_notes": "notes",
    "implementation_notes": "notes",
    "commits_summary": "commits",
    "body": "description",
    "content": "description",
}

MARKDOWN_HEADING_RE = re.compile(r"^(?:#{1,6}\s+(.+?)\s*#*|h[1-6]\.\s+(.+?)|\*\*(.+?)\*\*)\s*$")
TITLE_LINE_RE = re.compile(r"^([A-Z][A-Za-z0-9 /&()'’,.-]{1,70}?):?\s*$")
BOLD_LABEL_RE = re.compile(r"\*\*\s*([A-Za-z][A-Za-z /&-]{1,40}?)\s*:\s*\*\*\s*(.*)$")
PLAIN_LABELS = (
    "owner", "owners", "author", "authors", "maintainer", "maintainers", "dri", "point of contact",
    "primary users", "stakeholders", "audience", "reviewers", "contacts", "contact", "team",
    "slack channel", "slack channels", "channels", "service name", "on-call", "oncall",
)
PLAIN_LABEL_RE = re.compile(
    r"^(" + "|".join(re.escape(l) for l in PLAIN_LABELS) + r")\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE
)
BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*)$")


def canonical_key(key: str) -> str:
    return KEY_SYNONYMS.get(key, key)


def split_pseudo_yaml(content: str) -> "OrderedDict[str, str]":
    """
    Split "key:\\nvalue..." blocks (keys at column 0, lowercase snake_case). Keys are an open
    vocabulary; synonyms are folded with `canonical_key`. Text before the first key is stored
    under "_preamble". Returns an empty dict when the content has no such keys.
    """
    sections: "OrderedDict[str, List[str]]" = OrderedDict()
    current = "_preamble"
    for line in (content or "").splitlines():
        m = PSEUDO_YAML_KEY_RE.match(line)
        if m:
            current = canonical_key(m.group(1))
            rest = line[m.end():].strip()
            if current in sections:
                sections[current].append("")
            else:
                sections[current] = []
            if rest:
                sections[current].append(rest)
            continue
        sections.setdefault(current, []).append(line)
    out: "OrderedDict[str, str]" = OrderedDict()
    for key, lines in sections.items():
        text = "\n".join(lines).strip()
        if key == "_preamble" and not text:
            continue
        out[key] = text
    if list(out.keys()) in ([], ["_preamble"]):
        return OrderedDict()
    return out


def split_markdown_sections(content: str) -> List[Tuple[str, str]]:
    """Return [(heading, body)] using markdown / jira-wiki headings and short Title-Case label lines."""
    sections: List[Tuple[str, List[str]]] = [("", [])]
    lines = (content or "").splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        heading = None
        m = MARKDOWN_HEADING_RE.match(stripped)
        if m:
            heading = next(g for g in m.groups() if g)
        elif TITLE_LINE_RE.match(stripped) and len(stripped.split()) <= 8 and not BULLET_RE.match(line):
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            prev_line = lines[i - 1].strip() if i > 0 else ""
            if (next_line == "" or prev_line == "" or stripped.endswith(":")) and not stripped.endswith("."):
                heading = stripped.rstrip(":")
        if heading is not None:
            sections.append((heading.strip(), []))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(b).strip()) for h, b in sections if h or "\n".join(b).strip()]


def bold_label_values(content: str) -> Dict[str, str]:
    """
    Collect "**Owners:** value" / "Owners: value" metadata lines. When the value is empty the
    following bullet lines are joined with "; ". Keys are lower-cased labels.
    """
    values: Dict[str, str] = {}
    lines = (content or "").splitlines()

    def collect_bullets(start: int) -> str:
        items = []
        for j in range(start, min(start + 12, len(lines))):
            m = BULLET_RE.match(lines[j])
            if not m:
                if lines[j].strip() == "" and not items:
                    continue
                break
            items.append(m.group(1).strip())
        return "; ".join(items)

    for i, line in enumerate(lines):
        stripped = line.strip()
        m = BOLD_LABEL_RE.search(stripped)
        if m:
            label, value = m.group(1).strip().lower(), m.group(2).strip()
        else:
            m = PLAIN_LABEL_RE.match(stripped)
            if not m:
                continue
            label, value = m.group(1).strip().lower(), m.group(2).strip()
        if not value:
            value = collect_bullets(i + 1)
        if value and label not in values:
            values[label] = value
    return values


def split_person_list(value: str, require_full_name: bool = True) -> List[str]:
    """
    "Vanessa Ortiz; Runtime TL: Noah Patel; SRE: Sean Gallagher / Rafael Mendes" ->
    ["Vanessa Ortiz", "Noah Patel", "Sean Gallagher", "Rafael Mendes"].
    """
    names = []
    for part in re.split(r"[;,/|&]|\band\b|\n", value or ""):
        part = part.strip().strip("-•* ")
        if not part:
            continue
        if ":" in part:
            part = part.rsplit(":", 1)[1].strip()
        name = display_name(part)
        if not name:
            continue
        if require_full_name and not looks_like_full_name(name):
            continue
        names.append(name)
    return names
