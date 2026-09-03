"""
Structured query understanding: intent, keywords, entities, people, projects, ticket keys, a time
window and source-type constraints extracted from a question.

Two extractors feed one `QueryConstraints`:
- rules (offline, deterministic): src/metadata/dates.py + query_time_window, the project
  vocabulary, ticket-key regex, name regex and a curated source-type keyword map;
- one LLM call (query_name, JSON output, cached on disk per question) that is merged with the
  rules: lists are unioned after normalisation, the LLM time window wins when it parses.
`query_understanding` = "none" | "rules" | "llm" selects the behaviour.
"""
import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..metadata.dates import normalize_date
from ..metadata.document import unique
from ..metadata.parsers import ENTERPRISE_SOURCE_TYPES
from ..metadata.patterns import HASH_CHANNEL_RE, display_name, extract_ticket_keys, find_full_names
from ..metadata.vocab import ProjectVocabulary
from ..prompt.rag_query import PROMPT_VERSION, get_query_template, parse_query_response
from .time_expressions import query_time_window

INTENTS = ("lookup", "aggregation", "comparison", "timeline", "unknown")

# Only explicit system names map to a source type. Content words ("ticket", "meeting", "channel",
# "pull request", "deal", "runbook") are NOT used: EnterpriseRAG-Bench questions describe the
# information, not the system it lives in, and those words mislead more often than they help.
SOURCE_KEYWORDS: Dict[str, Sequence[str]] = {
    "slack": (r"\bslack\b", r"#[a-z][a-z0-9-]{1,30}\b"),
    "gmail": (r"\bgmail\b", r"\be-?mails?\b", r"\bemail thread\b", r"\bmail thread\b"),
    "fireflies": (r"\bfireflies\b",),
    "jira": (r"\bjira\b",),
    "linear": (r"\blinear (?:issue|ticket)s?\b", r"\bin linear\b"),
    "github": (r"\bgithub\b",),
    "confluence": (r"\bconfluence\b",),
    "google_drive": (r"\bgoogle (?:drive|docs?|sheets?|slides?)\b", r"\bgdrive\b"),
    "hubspot": (r"\bhubspot\b",),
}
_SOURCE_RES = {src: re.compile("|".join(f"(?:{p})" for p in pats), re.IGNORECASE)
               for src, pats in SOURCE_KEYWORDS.items()}
_PERSON_HINT_RE = re.compile(
    r"(?:\b(?:by|from|with|between|according to|asked|said|wrote|reported|assigned to|owned by|"
    r"cc'?d|replied|per|via|did|does|has|have|is|was|were|told|mentioned|when|what|whom|who)\s+)"
    r"([A-Z][^\W\d_]+(?:\s+[A-Z][^\W\d_]+){1,2})"
)


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _clean_list(values: Iterable, cap: int = 20) -> List[str]:
    out = []
    for v in values or []:
        if isinstance(v, (list, tuple)):
            out.extend(_clean_list(v, cap))
            continue
        if v is None:
            continue
        s = re.sub(r"\s+", " ", str(v)).strip().strip("\"'“”‘’#")
        if s and s.lower() not in ("null", "none", "n/a", "unknown"):
            out.append(s)
    return unique(out, cap=cap)


def _merge_lists(*groups: Iterable[str], cap: int = 20) -> List[str]:
    seen = {}
    for group in groups:
        for v in group or []:
            key = _norm(v)
            if key and key not in seen:
                seen[key] = v
    return list(seen.values())[:cap]


@dataclass
class QueryConstraints:
    intent: str = "unknown"
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    people: List[str] = field(default_factory=list)
    projects: List[str] = field(default_factory=list)
    ticket_keys: List[str] = field(default_factory=list)
    time_range: Optional[Dict[str, str]] = None          # {"start": ISO date, "end": ISO date}, inclusive
    source_types: List[str] = field(default_factory=list)
    channels: List[str] = field(default_factory=list)
    method: str = "none"                                  # none | rules | llm+rules | llm
    raw_llm: Optional[Dict[str, Any]] = None

    FIELDS = ("source_types", "time", "people", "projects", "entities", "ticket_keys", "channels")

    def has_constraints(self) -> bool:
        return bool(self.entities or self.people or self.projects or self.ticket_keys
                    or self.time_range or self.source_types or self.channels)

    def specified_fields(self) -> List[str]:
        out = []
        if self.source_types:
            out.append("source_type")
        if self.time_range:
            out.append("time")
        for name in ("people", "projects", "entities", "ticket_keys", "channels"):
            if getattr(self, name):
                out.append(name)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryConstraints":
        data = dict(data or {})
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def empty(cls) -> "QueryConstraints":
        return cls()


# ----------------------------------------------------------------------------- rules -----------

def _time_range_from(start: Optional[str], end: Optional[str]) -> Optional[Dict[str, str]]:
    s = normalize_date(start) if start else None
    e = normalize_date(end) if end else None
    if s is None and e is None:
        return None
    s = (s or e)[:10]
    e = (e or s)[:10]
    if s > e:
        s, e = e, s
    return {"start": s, "end": e}


def detect_source_types(text: str) -> List[str]:
    found = []
    for src, pattern in _SOURCE_RES.items():
        if pattern.search(text or ""):
            found.append(src)
    return found


def parse_query_rules(question: str, vocab: Optional[ProjectVocabulary] = None) -> QueryConstraints:
    """Deterministic extraction (no model): dates, source words, ticket keys, vocabulary, names."""
    text = question or ""
    projects, entities = vocab.match(text) if vocab is not None else ([], [])
    ticket_keys = unique(extract_ticket_keys(text))
    window = query_time_window(text)
    people = []
    for m in _PERSON_HINT_RE.finditer(text):
        people.extend(find_full_names(m.group(1)))
    channels = [c for c in HASH_CHANNEL_RE.findall(text)]
    return QueryConstraints(
        intent="unknown",
        keywords=[],
        entities=unique(list(entities) + [k for k in ticket_keys if k not in entities]),
        people=unique(people),
        projects=unique(projects),
        ticket_keys=ticket_keys,
        time_range={"start": window[0], "end": window[1]} if window else None,
        source_types=detect_source_types(text),
        channels=unique(channels),
        method="rules",
    )


# ------------------------------------------------------------------------------ LLM ------------

def normalize_llm_output(data: Dict[str, Any], vocab: Optional[ProjectVocabulary] = None) -> QueryConstraints:
    """Coerce the model JSON into a QueryConstraints (unknown keys dropped, values sanitised)."""
    intent = _norm(data.get("intent")) if isinstance(data.get("intent"), str) else "unknown"
    if intent not in INTENTS:
        intent = "unknown"
    tr = data.get("time_range")
    time_range = None
    if isinstance(tr, dict):
        time_range = _time_range_from(tr.get("start"), tr.get("end"))
    elif isinstance(tr, str):
        time_range = _time_range_from(tr, tr)
    source_types = [s for s in (_norm(x).replace(" ", "_") for x in _clean_list(data.get("source_types")))
                    if s in ENTERPRISE_SOURCE_TYPES]
    people = [display_name(p) for p in _clean_list(data.get("people"))]
    people = [p for p in people if p and len(p.split()) <= 4]
    ticket_keys = unique(extract_ticket_keys(" ".join(_clean_list(data.get("ticket_keys")))))
    projects = _clean_list(data.get("projects"))
    entities = _clean_list(data.get("entities"))
    if vocab is not None and not vocab.is_empty():
        # map free-form names onto canonical vocabulary names when they match
        canon_p, canon_e = vocab.match(" ; ".join(projects + entities))
        projects = _merge_lists(canon_p, projects)
        entities = _merge_lists(canon_e, entities)
    return QueryConstraints(
        intent=intent,
        keywords=_clean_list(data.get("keywords"), cap=12),
        entities=entities,
        people=unique(people),
        projects=projects,
        ticket_keys=ticket_keys,
        time_range=time_range,
        source_types=unique(source_types),
        channels=_clean_list(data.get("channels")),
        method="llm",
        raw_llm=data,
    )


def merge_constraints(llm: Optional[QueryConstraints], rules: QueryConstraints) -> QueryConstraints:
    """Union of both extractors; the LLM window wins when present, otherwise the rules window."""
    if llm is None:
        return rules
    return QueryConstraints(
        intent=llm.intent if llm.intent != "unknown" else rules.intent,
        keywords=_merge_lists(llm.keywords, rules.keywords, cap=12),
        entities=_merge_lists(llm.entities, rules.entities),
        people=_merge_lists(llm.people, rules.people),
        projects=_merge_lists(llm.projects, rules.projects),
        ticket_keys=_merge_lists(llm.ticket_keys, rules.ticket_keys),
        time_range=llm.time_range or rules.time_range,
        source_types=_merge_lists(llm.source_types, rules.source_types),
        channels=_merge_lists(llm.channels, rules.channels),
        method="llm+rules",
        raw_llm=llm.raw_llm,
    )


def vocabulary_for_tree(conf: Dict, tree=None) -> ProjectVocabulary:
    """
    The vocabulary used at index time: conf["enterprise_project_vocab"] plus HubSpot account
    titles of the indexed documents (mirrors DataManager._preprocess_enterprise).
    """
    vocab = ProjectVocabulary.load(conf.get("enterprise_project_vocab"))
    documents = getattr(tree, "documents", None) or {}
    accounts = {}
    for doc in documents.values():
        title = (doc.title or "").strip()
        if doc.source_type == "hubspot" and 3 <= len(title) <= 60:
            accounts[title] = [title]
    if accounts:
        vocab = ProjectVocabulary(vocab.projects, {**accounts, **vocab.entities})
    return vocab


class QueryUnderstanding:
    """Parse questions into QueryConstraints according to conf["query_understanding"]."""

    def __init__(self, conf: Dict, vocab: Optional[ProjectVocabulary] = None, model=None,
                 cache_path: Optional[str] = None) -> None:
        self.conf = conf
        self.mode = str(conf.get("query_understanding") or "none").lower()
        if self.mode not in ("none", "rules", "llm"):
            raise ValueError(f'query_understanding must be "none", "rules" or "llm", got "{self.mode}".')
        self.vocab = vocab if vocab is not None else ProjectVocabulary.load(conf.get("enterprise_project_vocab"))
        self.model = model
        self.model_name = None
        self._lock = threading.Lock()
        self._cache: Dict[str, Dict] = {}
        self.cache_path: Optional[str] = None
        self.stats = {"llm_calls": 0, "cache_hits": 0, "parse_failures": 0}
        if self.mode == "llm":
            self.model_name = conf.get("query_name") or conf.get("qa_name")
            if self.model is None:
                self.model = conf.get("query_model")
            if self.model is None and self.model_name:
                from ..model.factory import build_model   # lazy: src.model pulls torch / transformers
                self.model = build_model(self.model_name, "query", conf)
            if self.model is None:
                raise ValueError('query_understanding="llm" needs query_name (or qa_name) to be set.')
            self.cache_path = cache_path or self._default_cache_path()
            self._load_cache()

    # ---- cache -------------------------------------------------------------------------------
    def _default_cache_path(self) -> Optional[str]:
        cache_dir = self.conf.get("query_cache_dir")
        if cache_dir is None:
            save_dir = self.conf.get("save_dir")
            if not save_dir:
                return None
            cache_dir = os.path.join(save_dir, "query_cache")
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(self.model_name))
        return os.path.join(cache_dir, f"{name}_{PROMPT_VERSION}.json")

    def _load_cache(self) -> None:
        if self.cache_path and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except (OSError, json.JSONDecodeError):
                logging.warning(f'Query cache "{self.cache_path}" unreadable; starting empty.')
                self._cache = {}

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False)
        os.replace(tmp, self.cache_path)

    @staticmethod
    def _key(question: str) -> str:
        return hashlib.sha1(question.strip().encode("utf-8")).hexdigest()

    # ---- parsing -----------------------------------------------------------------------------
    def _llm_json(self, question: str) -> Optional[Dict[str, Any]]:
        key = self._key(question)
        with self._lock:
            if key in self._cache:
                self.stats["cache_hits"] += 1
                return self._cache[key].get("json")
        hint = list(self.vocab.projects.keys())[:60] if self.vocab is not None else None
        messages = get_query_template(question, ENTERPRISE_SOURCE_TYPES, project_hint=hint)
        try:
            response = self.model.qa(messages, max_tokens=400)
        except Exception as e:   # the OpenAI wrapper already retries; never fail retrieval on parsing
            logging.exception("query understanding call failed")
            response = None
        data = parse_query_response(response if isinstance(response, str) else None)
        with self._lock:
            self.stats["llm_calls"] += 1
            if data is None:
                self.stats["parse_failures"] += 1
            self._cache[key] = {"question": question, "json": data,
                                "raw": response if isinstance(response, str) else None}
            self._save_cache()
        return data

    def parse(self, question: str) -> QueryConstraints:
        if self.mode == "none":
            return QueryConstraints.empty()
        rules = parse_query_rules(question, self.vocab)
        if self.mode == "rules":
            return rules
        data = self._llm_json(question)
        llm = normalize_llm_output(data, self.vocab) if isinstance(data, dict) else None
        return merge_constraints(llm, rules)
