"""Markdown-ish documents: confluence pages and google_drive files."""
import re
from typing import Dict, List

from ..dates import normalize_date
from ..document import Document, DocumentMetadata, unique
from ..patterns import HASH_CHANNEL_RE, find_full_names
from ..sections import bold_label_values, split_markdown_sections, split_person_list
from .base import BaseMetadataParser

AUTHOR_LABELS = ("owner", "owners", "author", "authors", "maintainer", "maintainers", "dri",
                 "point of contact", "doc owner", "document owner", "tech lead", "lead")
PARTICIPANT_LABELS = ("primary users", "stakeholders", "audience", "reviewers", "contacts", "contact",
                      "team", "on-call", "oncall", "approvers", "consumers", "users")
CHANNEL_LABELS = ("slack channel", "slack channels", "channels", "channel")
DATE_RANGE_RE = re.compile(
    r"\bDates?\s*:\s*(\d{4}-\d{2}-\d{2})(?:\s*(?:to|–|-|through)\s*(\d{4}-\d{2}-\d{2}))?", re.IGNORECASE
)
OWNER_LINE_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?\**\s*([A-Za-z /]*(?:owner|owners|dri|maintainer|author|point of contact|poc|tech lead|lead|approver)s?)\s*\**\s*:\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
CONTACT_LINE_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?\**\s*((?:primary|secondary|backup)?\s*(?:contact|contacts|stakeholders|reviewers|escalation|on-?call|pager|sme|smes))\s*\**\s*:\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
UPDATED_RE = re.compile(r"\b(?:last updated|updated|revised|last reviewed)\s*[:\-–]?\s*([A-Za-z0-9,/ -]{6,30})", re.IGNORECASE)


class MarkdownishParser(BaseMetadataParser):
    source_type = "wiki"

    def _parse_metadata(self, md: DocumentMetadata, title: str, content: str) -> None:
        labels = bold_label_values(content)
        authors, participants, channels = [], [], []
        for label, value in labels.items():
            if label in AUTHOR_LABELS:
                authors.extend(split_person_list(value))
            elif label in PARTICIPANT_LABELS:
                participants.extend(split_person_list(value))
            elif label in CHANNEL_LABELS:
                channels.extend(HASH_CHANNEL_RE.findall(value))
            elif label == "service name" and len(value) <= 60:
                md.extra["service_name"] = value
        for label, value in OWNER_LINE_RE.findall(content or ""):
            authors.extend(find_full_names(value))
        for label, value in CONTACT_LINE_RE.findall(content or ""):
            participants.extend(find_full_names(value))
        for label, value in labels.items():
            if label in AUTHOR_LABELS:
                authors.extend(find_full_names(value))
        channels = unique(channels + HASH_CHANNEL_RE.findall(content or ""), 10)
        md.authors = unique(authors)
        md.participants = unique(participants)
        md.entities = [f"#{c}" for c in channels]
        md.extra["slack_channels"] = channels
        md.extra["labels"] = {k: v[:200] for k, v in list(labels.items())[:20]}

        headings = [h for h, _ in split_markdown_sections(content) if h]
        md.extra["sections"] = unique(headings, 40)

        m = DATE_RANGE_RE.search(content or "")
        if m:
            md.created_at = normalize_date(m.group(1))
            md.updated_at = normalize_date(m.group(2)) if m.group(2) else md.created_at
        um = UPDATED_RE.search(content or "")
        if um:
            iso = normalize_date(um.group(1).strip())
            if iso:
                md.updated_at = max(md.updated_at, iso) if md.updated_at else iso

    def local_metadata(self, chunk_text: str, doc: Document, state: Dict) -> Dict:
        headings: List[str] = doc.metadata.extra.get("sections", [])
        last_pos, last_heading = -1, None
        for heading in headings:
            if len(heading) < 3:
                continue
            pos = chunk_text.rfind(heading)
            if pos > last_pos:
                last_pos, last_heading = pos, heading
        if last_heading is not None:
            state["section"] = last_heading
        return {"section": state.get("section")}


class ConfluenceParser(MarkdownishParser):
    source_type = "confluence"


class GoogleDriveParser(MarkdownishParser):
    source_type = "google_drive"
