"""Base class shared by the per-source metadata parsers (rules only, no LLM)."""
import re
from typing import Dict, List, Optional

from ..dates import date_bounds, find_dates
from ..document import Document, DocumentMetadata, TimeRange, unique
from ..patterns import EMAIL_RE, extract_ticket_keys
from ..vocab import ProjectVocabulary

LIST_CAPS = {
    "authors": 20, "participants": 40, "projects": 20, "entities": 40,
    "ticket_keys": 40, "emails": 40, "dates_mentioned": 20,
}


class BaseMetadataParser:
    source_type: str = "generic"

    # ------------------------------------------------------------------ public API
    def normalize_content(self, content: str) -> str:
        """Turn the raw `content` column into plain text (gmail / hubspot override this)."""
        return content or ""

    def parse(self, doc_id: str, source_type: str, title: str, content: str,
              vocab: Optional[ProjectVocabulary] = None) -> Document:
        title = title or ""
        content = content or ""
        md = DocumentMetadata()
        self._parse_metadata(md, title, content)
        self._finalize(md, title, content, vocab)
        return Document(document_id=doc_id, title=title, source_type=source_type,
                        metadata=md, num_chars=len(content))

    def local_metadata(self, chunk_text: str, doc: Document, state: Dict) -> Dict:
        """
        Cheap per-chunk metadata. `state` is carried over between consecutive chunks of the
        same document so section / speaker information survives chunk boundaries. Chunks are
        space-joined (no newlines), so implementations must not rely on line anchors.
        """
        return {}

    # ------------------------------------------------------------------ hooks
    def _parse_metadata(self, md: DocumentMetadata, title: str, content: str) -> None:
        pass

    def _finalize(self, md: DocumentMetadata, title: str, content: str,
                  vocab: Optional[ProjectVocabulary]) -> None:
        text = f"{title}\n{content}"
        md.ticket_keys = unique(md.ticket_keys + extract_ticket_keys(text), LIST_CAPS["ticket_keys"])
        md.emails = unique(md.emails + EMAIL_RE.findall(text), LIST_CAPS["emails"])
        if vocab is not None:
            projects, entities = vocab.match(text)
            md.projects = unique(md.projects + projects)
            md.entities = unique(md.entities + entities)
        md.entities = unique(md.entities + md.ticket_keys, LIST_CAPS["entities"])
        md.projects = unique(md.projects, LIST_CAPS["projects"])

        header_dates = [d for d in (md.created_at, md.updated_at) if d]
        dates = sorted(set(md.dates_mentioned) | set(header_dates) | set(find_dates(content)))
        md.dates_mentioned = dates[: LIST_CAPS["dates_mentioned"]]
        start, end = date_bounds(dates)
        if md.created_at is None and start is not None:
            md.created_at = start
            md.extra.setdefault("created_at_source", "inline")
        elif md.created_at is not None:
            md.extra.setdefault("created_at_source", "header")
        if md.updated_at is None and end is not None:
            md.updated_at = end
        if start is not None or end is not None:
            md.time_range = TimeRange(start, end)

        md.authors = unique(md.authors, LIST_CAPS["authors"])
        md.participants = unique(md.authors + md.participants, LIST_CAPS["participants"])

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def speakers_in_chunk(chunk_text: str, candidates: List[str]) -> List[str]:
        """Names from `candidates` that appear as a speaker prefix ("Name:" / "Name (role):") in the chunk."""
        found = []
        for name in candidates:
            if not name:
                continue
            pattern = rf"(?:^|\s){re.escape(name)}(?: \([^)]{{0,40}}\))?:"
            if re.search(pattern, chunk_text):
                found.append(name)
        return found

    @staticmethod
    def current_section(chunk_text: str, section_keys: List[str], state: Dict,
                        state_key: str = "section") -> Optional[str]:
        """Last known section key mentioned as "key:" in the chunk (carried over via `state`)."""
        last_pos, last_key = -1, None
        for key in section_keys:
            for m in re.finditer(rf"(?:^|\s){re.escape(key)}:", chunk_text):
                if m.start() > last_pos:
                    last_pos, last_key = m.start(), key
        if last_key is not None:
            state[state_key] = last_key
        return state.get(state_key)
