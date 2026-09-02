"""Gmail threads: one document = one or more messages with From/To/Cc/Date/Subject headers."""
import ast
import json
import re
from typing import Dict, List

from ..dates import normalize_date
from ..document import Document, DocumentMetadata, unique
from ..patterns import EMAIL_RE, parse_address_list
from .base import BaseMetadataParser

HEADER_RE = re.compile(r"^(From|To|Cc|Bcc|Date|Sent|Subject|Attachments?|Attached):\s*(.*)$", re.IGNORECASE)
MESSAGE_SPLIT_RE = re.compile(r"(?m)^(?=From:\s)")
SUBJECT_PREFIX_RE = re.compile(r"^(?:(?:re|fwd?|fw)\s*:\s*)+", re.IGNORECASE)
INTERNAL_DOMAIN = "redwoodinference.com"


def _unescape(text: str) -> str:
    if "\\n" in text:
        text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
        text = text.replace("\\'", "'").replace('\\"', '"')
    return text


class GmailParser(BaseMetadataParser):
    source_type = "gmail"

    def normalize_content(self, content: str) -> str:
        raw = (content or "").strip()
        messages = None
        if raw[:1] in ("[", "("):
            for loader in (ast.literal_eval, json.loads):
                try:
                    obj = loader(raw)
                except Exception:
                    continue
                if isinstance(obj, (list, tuple)) and obj and all(isinstance(m, str) for m in obj):
                    messages = [m.strip() for m in obj]
                    break
        text = "\n\n".join(messages) if messages is not None else raw
        return _unescape(text)

    @staticmethod
    def split_messages(content: str) -> List[str]:
        parts = [p.strip() for p in MESSAGE_SPLIT_RE.split(content or "") if p.strip()]
        return parts or ([content.strip()] if content and content.strip() else [])

    @staticmethod
    def parse_headers(message: str) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        current = None
        for line in message.splitlines()[:40]:
            if not line.strip():
                if headers:
                    break
                continue
            m = HEADER_RE.match(line)
            if m:
                current = m.group(1).lower()
                if current in ("attachment", "attached"):
                    current = "attachments"
                if current == "sent":
                    current = "date"
                headers[current] = (headers.get(current, "") + " " + m.group(2).strip()).strip()
            elif current and line[:1].isspace():
                headers[current] = (headers[current] + " " + line.strip()).strip()
            elif headers:
                break
        return headers

    def _parse_metadata(self, md: DocumentMetadata, title: str, content: str) -> None:
        messages = self.split_messages(content)
        authors, participants, dates, attachments, subjects = [], [], [], [], []
        external_domains = []
        for message in messages:
            headers = self.parse_headers(message)
            if "from" in headers:
                names = parse_address_list(headers["from"])
                if names:
                    authors.append(names[0])
            for key in ("to", "cc", "bcc"):
                if key in headers:
                    participants.extend(parse_address_list(headers[key]))
            if "date" in headers:
                iso = normalize_date(headers["date"])
                if iso:
                    dates.append(iso)
            if "subject" in headers:
                subjects.append(SUBJECT_PREFIX_RE.sub("", headers["subject"]).strip())
            if "attachments" in headers:
                attachments.extend(a.strip() for a in re.split(r"[;,]", headers["attachments"]) if a.strip())
            for email in EMAIL_RE.findall(" ".join(headers.get(k, "") for k in ("from", "to", "cc", "bcc"))):
                domain = email.split("@", 1)[1].lower()
                if INTERNAL_DOMAIN not in domain:
                    external_domains.append(domain)

        md.authors = unique(authors)
        md.participants = unique(participants)
        if dates:
            md.created_at, md.updated_at = min(dates), max(dates)
            md.dates_mentioned = sorted(set(dates))
        md.extra.update({
            "num_messages": len(messages),
            "subject": subjects[0] if subjects else (title or None),
            "attachments": unique(attachments, 20),
            "external_domains": unique(external_domains, 10),
        })
        md.entities = unique(external_domains, 10)

    def local_metadata(self, chunk_text: str, doc: Document, state: Dict) -> Dict:
        for m in re.finditer(r"From:\s*(.+?)(?=\s+(?:To|Cc|Date|Subject):|$)", chunk_text):
            state["message_index"] = state.get("message_index", 0) + 1
            names = parse_address_list(m.group(1))
            if names:
                state["from"] = names[0]
        return {"message_index": state.get("message_index", 1), "from": state.get("from")}
