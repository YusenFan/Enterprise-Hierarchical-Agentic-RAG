"""Pseudo-YAML sources: jira, linear, github (PRs) and hubspot (CRM records)."""
import re
from typing import Dict

from ..dates import normalize_date
from ..document import Document, DocumentMetadata, unique
from ..patterns import ORG_WORDS, SPEAKER_STOPLIST, find_full_names
from ..sections import split_person_list, split_pseudo_yaml
from .base import BaseMetadataParser

AUTHOR_KEYS = ("reporter", "author", "authors", "owner", "owners", "requester", "requested_by",
               "created_by", "submitted_by", "opened_by", "account_owner", "account_manager", "ae", "se",
               "sales_rep", "csm", "champion")
PARTICIPANT_KEYS = ("assignee", "assignees", "reviewer", "reviewers", "participants", "attendees",
                    "stakeholders", "contacts", "contact", "cc", "watchers", "collaborators",
                    "decision_maker", "decision_makers", "technical_contact", "business_contact")
PROJECT_KEYS = ("project", "projects", "initiative", "epic", "workstream", "program")
SHORT_VALUE_KEYS = ("status", "priority", "severity", "labels", "type", "issue_type", "repo",
                    "repository", "component", "components", "environment", "team", "customer",
                    "account", "company", "stage", "deal_stage", "region", "tier", "plan", "branch",
                    "milestone", "sprint", "version", "product", "category")
CREATED_KEYS = ("created", "created_at", "opened", "opened_at", "date", "reported", "reported_at", "start_date")
UPDATED_KEYS = ("updated", "updated_at", "resolved", "resolved_at", "closed", "closed_at", "merged",
                "merged_at", "due", "due_date", "last_activity", "last_updated", "end_date")
TITLE_CASE_SUBLABEL_RE = re.compile(r"^([A-Z][A-Za-z /()&-]{2,50}?):\s*(.*)$", re.MULTILINE)
REVIEW_RUN_RE = re.compile(r"(?:^|(?<=[.!?)\]])\s+)([A-Z][a-z]+(?: [A-Z][a-z'’-]+){1,2}):\s")
SUBLABEL_PEOPLE = {"reporter", "assignee", "owner", "requester", "customer contact", "reported by"}
SUBLABEL_SHORT = {"customer", "account", "priority", "severity", "status", "component", "environment",
                  "region", "tier", "deployment", "product"}


COMMENT_KEYS = ("comments", "review_comments", "investigation_notes", "updates", "progress_updates",
                "decision_log", "review_feedback", "discussion", "thread", "activity", "worklog")
# "Marco: text. Priya (author): text." -> split before a capitalised name followed by ":"
COMMENT_RUN_SPLIT_RE = re.compile(r"(?<=[.!?\]])\s+(?=[A-Z][\w'’-]+(?: [A-Z][\w'’-]+){0,2}(?: \([^)]{0,40}\))?:\s)")
COMMENT_PREFIX_RE = re.compile(r"^(.{0,100}?)(?::\s|:$)")
SINGLE_NAME_RE = re.compile(r"^(?:\d{4}-\d{2}-\d{2}[^A-Za-z(]*)?([A-Z][a-z]{2,15})(?:\s*\([^)]*\))?$")
ROLE_RE = re.compile(r"\(([^)]{1,40})\)")
AUTHOR_ROLES = {"author", "reporter", "requester", "owner", "opened by"}


def extract_commenters(block: str):
    """
    Names that open comment entries in jira / linear / github blocks, e.g.
    "2026-03-10 Maya Chen: ...", "2025-05-07 (Liam O'Rourke, Support): ...",
    "Priya Kapoor (author): ...", "Marco: ...". Returns (names, roles).
    """
    names, roles = [], {}
    segments = []
    for line in (block or "").splitlines():
        segments.extend(COMMENT_RUN_SPLIT_RE.split(line))
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        m = COMMENT_PREFIX_RE.match(segment)
        if m:
            prefix = m.group(1)
        elif re.match(r"^\d{4}-\d{2}-\d{2}", segment):
            # "2025-05-19 10:14 PT — Jasmine Liu (Reporter)": a header line, scan it whole
            prefix = segment[:100]
        elif re.search(r"^.{0,60}?\s[—–]\s", segment):
            # "Aisha Kline (Support) 2026-03-10 02:20 — text": keep the part before the dash
            prefix = re.split(r"\s[—–]\s", segment, maxsplit=1)[0]
        else:
            continue  # body prose: never scanned for names
        found = find_full_names(prefix)
        if not found:
            sm = SINGLE_NAME_RE.match(prefix.strip())
            if sm and sm.group(1).lower() not in ORG_WORDS and sm.group(1).lower() not in SPEAKER_STOPLIST:
                found = [sm.group(1)]
        role_match = ROLE_RE.search(prefix)
        role = role_match.group(1).strip() if role_match and not re.search(r"\d{4}", role_match.group(1)) else None
        for name in found:
            names.append(name)
            if role and name not in roles and name not in role:
                roles[name] = role
    return names, roles


class PseudoYamlParser(BaseMetadataParser):
    source_type = "ticket"

    def _parse_metadata(self, md: DocumentMetadata, title: str, content: str) -> None:
        sections = split_pseudo_yaml(content)
        md.extra["sections"] = [k for k in sections.keys() if k != "_preamble"]
        authors, participants, projects = [], [], []
        for key, value in sections.items():
            short = value if len(value) <= 300 and "\n" not in value.strip() else None
            if key in AUTHOR_KEYS and short:
                authors.extend(split_person_list(short, require_full_name=False))
            elif key in PARTICIPANT_KEYS and short:
                participants.extend(split_person_list(short, require_full_name=False))
            elif key in PROJECT_KEYS and short:
                projects.extend(p.strip() for p in re.split(r"[;,]", short) if p.strip())
            elif key in SHORT_VALUE_KEYS and short:
                md.extra[key] = short
            elif key in CREATED_KEYS and short:
                iso = normalize_date(short)
                if iso and md.created_at is None:
                    md.created_at = iso
            elif key in UPDATED_KEYS and short:
                iso = normalize_date(short)
                if iso:
                    md.updated_at = max(md.updated_at, iso) if md.updated_at else iso
        commenters, roles = [], {}
        for key in COMMENT_KEYS:
            if key in sections:
                found, found_roles = extract_commenters(sections[key])
                commenters.extend(found)
                roles.update(found_roles)
        for name, role in roles.items():
            if role.lower() in AUTHOR_ROLES:
                authors.append(name)
        participants.extend(commenters)
        if roles:
            md.extra["roles"] = roles
        md.authors = unique(authors)
        md.participants = unique(participants)
        md.projects = unique(projects)
        self._parse_source_specific(md, title, content, sections)

    def _parse_source_specific(self, md: DocumentMetadata, title: str, content: str, sections: Dict[str, str]) -> None:
        pass

    def local_metadata(self, chunk_text: str, doc: Document, state: Dict) -> Dict:
        keys = doc.metadata.extra.get("sections", [])
        return {"section": self.current_section(chunk_text, keys, state)}


class JiraParser(PseudoYamlParser):
    source_type = "jira"

    def _parse_source_specific(self, md, title, content, sections) -> None:
        description = sections.get("description", "")
        sublabels = []
        for label, value in TITLE_CASE_SUBLABEL_RE.findall(description):
            label_l = label.strip().lower()
            sublabels.append(label.strip())
            value = value.strip()
            if not value:
                continue
            if label_l in SUBLABEL_PEOPLE:
                names = split_person_list(value)
                if label_l in ("reporter", "reported by", "requester"):
                    md.authors = unique(md.authors + names)
                else:
                    md.participants = unique(md.participants + names)
            elif label_l in SUBLABEL_SHORT and len(value) <= 120:
                md.extra.setdefault(label_l, value)
        md.extra["description_sections"] = unique(sublabels, 20)
        customer = md.extra.get("customer") or md.extra.get("account")
        if customer and len(customer) <= 60:
            md.entities = unique(md.entities + [customer])


class LinearParser(PseudoYamlParser):
    source_type = "linear"


class GithubParser(PseudoYamlParser):
    source_type = "github"

    def _parse_source_specific(self, md, title, content, sections) -> None:
        reviewers = []
        for key in ("review_comments", "comments"):
            if key in sections:
                reviewers.extend(extract_commenters(sections[key])[0])
        roles = md.extra.get("roles", {})
        reviewers = [r for r in reviewers if roles.get(r, "").lower() != "author"]
        md.participants = unique(md.participants + reviewers)
        md.extra["pr_title"] = title
        md.extra["reviewers"] = unique(reviewers, 20)


class HubspotParser(PseudoYamlParser):
    source_type = "hubspot"

    def normalize_content(self, content: str) -> str:
        text = content or ""
        if "\\n" in text:
            text = text.replace("\\n", "\n").replace("\\t", "\t")
        return text

    def _parse_source_specific(self, md, title, content, sections) -> None:
        account = title.strip()
        if account:
            md.extra["account"] = account
            md.entities = unique([account] + md.entities)
        timeline = sections.get("timeline", "")
        entries = re.findall(r"(?m)^\s*(\d{4}-\d{2}-\d{2})\s*(?:[:\-–]\s*)?(.*)$", timeline)
        if entries:
            md.extra["timeline_entries"] = len(entries)
            dates = sorted(normalize_date(d) for d, _ in entries if normalize_date(d))
            if dates:
                md.created_at = md.created_at or dates[0]
                md.updated_at = max(md.updated_at, dates[-1]) if md.updated_at else dates[-1]
