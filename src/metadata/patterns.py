"""Compiled regexes and small string helpers shared by the parsers."""
import re
from typing import List, Optional

TICKET_KEY_RE = re.compile(r"(?<![\w/#-])([A-Z]{2,6}-\d{1,6})(?![\w-])")
# Prefixes that look like ticket keys but are not (hash names, standards, hardware, units).
TICKET_KEY_STOPLIST = {
    "SHA", "UTF", "GPT", "ISO", "RFC", "AES", "RSA", "TLS", "SSL", "CVE", "HTTP", "UUID", "AWS",
    "GPU", "CPU", "API", "SDK", "SOC", "PCI", "IEEE", "CUDA", "DNS", "TCP", "UDP", "IPV", "MD",
    "VM", "GB", "MB", "TB", "KB", "FP", "BF", "UTC", "PST", "PDT", "EST", "EDT", "OS", "ID",
    "IP", "KV", "LLM", "ML", "AI", "TTL", "SLA", "SLO", "PII", "VPC", "IAM", "KMS", "SSO", "JWT",
    "MFA", "GDPR", "HIPAA", "CCPA", "EC", "US", "EU", "AP", "NA", "SA", "ARM", "X", "AMI",
}

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
CHANNEL_RE = re.compile(r"^[a-z][a-z0-9-]{1,30}$")
HASH_CHANNEL_RE = re.compile(r"#([a-z][a-z0-9-]{1,30})\b")
BOT_RE = re.compile(r"(?i)(?:^|[-_ ])bot$|[a-z]Bot$")
FULL_NAME_RE = re.compile(r"^[A-Z][^\W\d_'’.-]*(?:['’.-][^\W\d_]+)*(?: [A-Z][^\W\d_'’.-]*(?:['’.-][^\W\d_]+)*){1,3}$")
NAME_TOKEN_RE = re.compile(r"^[A-Za-z][\w'’.-]*(?: [A-Z][\w'’.-]*){0,3}$")

# "Sanaa (infra PM): text", "sasha: text", "IncidentBot: text" at the start of a line.
SLACK_SPEAKER_LINE_RE = re.compile(
    r"^([A-Za-z][\w.'’-]*(?: [A-Z][\w.'’-]*){0,2})(?: \(([^)]{1,40})\))?:\s", re.MULTILINE
)
# "[00:10] Sofia Alvarez: text"
FIREFLIES_UTTERANCE_RE = re.compile(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*([^:\n\[\]]{1,60}?):")

# Words that start lines with a colon but are not speakers.
SPEAKER_STOPLIST = {
    "note", "notes", "update", "updates", "summary", "links", "link", "action", "actions",
    "context", "repro", "steps", "impact", "status", "error", "stack", "thanks", "question",
    "answer", "ps", "re", "fwd", "subject", "from", "to", "cc", "date", "time", "duration",
    "attendees", "title", "location", "topics", "topic", "tl;dr", "tldr", "warning", "info",
    "debug", "trace", "result", "results", "example", "examples", "output", "input", "config",
    "log", "logs", "environment", "reminder", "checklist", "agenda", "decision", "decisions",
    "next", "todo", "background", "goal", "goals", "risk", "risks", "owner", "owners", "meeting",
    "transcript", "recording", "key", "keys", "eta", "cause", "fix", "mitigation", "root",
}


def extract_ticket_keys(text: str) -> List[str]:
    out = []
    for key in TICKET_KEY_RE.findall(text or ""):
        prefix = key.split("-", 1)[0]
        if prefix in TICKET_KEY_STOPLIST:
            continue
        out.append(key)
    return out


def name_from_email(email: str) -> Optional[str]:
    """first.last@x / first_last@x -> "First Last"."""
    local = (email or "").split("@", 1)[0]
    parts = [p for p in re.split(r"[._-]+", local) if p and not p.isdigit()]
    if not parts:
        return None
    return " ".join(p.capitalize() for p in parts)


def display_name(value: str) -> str:
    """'"Alice Tan" <alice@x>' / "Sanaa (infra PM)" / "Alice Tan:" -> "Alice Tan"."""
    value = (value or "").strip()
    value = re.sub(r"<[^>]*>", "", value)
    value = re.sub(r"\([^)]*\)", "", value)
    value = value.strip().strip("\"'“”‘’").rstrip(":").strip()
    value = re.sub(r"\s+", " ", value)
    if EMAIL_RE.fullmatch(value):
        return name_from_email(value) or value
    return value


def parse_address_list(value: str) -> List[str]:
    """Split a To/Cc header into display names (emails without a name become "First Last")."""
    names = []
    for part in re.split(r"[;,]", value or ""):
        part = part.strip()
        if not part:
            continue
        name = display_name(part)
        if not name:
            emails = EMAIL_RE.findall(part)
            name = name_from_email(emails[0]) if emails else ""
        if name:
            names.append(name)
    return names


ORG_WORDS = {
    "customer", "success", "eng", "engineering", "platform", "team", "runtime", "infra", "support",
    "sales", "security", "product", "ops", "sre", "onboarding", "manager", "marketing", "finance",
    "legal", "design", "docs", "data", "cloud", "service", "services", "solutions", "systems", "labs",
    "group", "network", "inc", "corp", "bank", "health", "capital", "partners", "console", "compliance",
    "enterprise", "private", "dedicated", "hosted", "release", "program", "project", "model", "models",
    "the", "and", "for", "with", "level", "tier", "primary", "secondary", "backup", "escalation",
    "oncall", "on-call", "pager", "owner", "owners", "lead", "leads", "manager", "managers", "api",
    "redwood", "inference", "speaker", "unknown", "customer", "vendor", "partner", "external", "internal",
    "action", "items", "next", "steps", "follow", "up", "summary", "notes", "meeting", "header",
}
_NAME_TOKEN = r"[A-Z][^\W\d_]*(?:['’-][^\W\d_]+)*"
FULL_NAME_FIND_RE = re.compile(rf"(?<![^\W\d_])({_NAME_TOKEN} {_NAME_TOKEN}(?: {_NAME_TOKEN})?)(?![^\W\d_])")


def find_full_names(text: str) -> List[str]:
    """Capitalised two/three-word names inside a short slot value, skipping org / role phrases."""
    names = []
    for candidate in FULL_NAME_FIND_RE.findall(text or ""):
        tokens = candidate.split()
        if any(t.lower().strip("'’-") in ORG_WORDS for t in tokens):
            continue
        if any(len(t) >= 2 and t.isupper() for t in tokens):   # "SRE Team", "GPU Burst"
            continue
        if any(len(t) < 2 for t in tokens):
            continue
        if candidate not in names:
            names.append(candidate)
    return names


def is_bot(name: str) -> bool:
    return bool(BOT_RE.search(name or ""))


def looks_like_full_name(value: str) -> bool:
    return bool(FULL_NAME_RE.match(value or ""))
