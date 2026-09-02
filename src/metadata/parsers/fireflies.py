"""Fireflies meeting transcripts: summary / transcript sections with a "Meeting Header" block."""
import re
from typing import Dict, List

from ..dates import combine_date_time, normalize_date
from ..document import Document, DocumentMetadata, unique
from ..patterns import FIREFLIES_UTTERANCE_RE, display_name, is_bot, looks_like_full_name
from ..sections import BULLET_RE, split_pseudo_yaml
from .base import BaseMetadataParser

MEETING_HEADER_RE = re.compile(r"^\s*Meeting Header:?\s*$", re.IGNORECASE | re.MULTILINE)
HEADER_LINE_RE = re.compile(
    r"^\s*(Date|Time|Start time|Start|Duration|Location|Recording|Title|Agenda|Organizer|Host|"
    r"Attendees(?:\s*\(([^)]*)\))?|Participants(?:\s*\(([^)]*)\))?)\s*:\s*(.*)$", re.IGNORECASE)
OTHER_HEADER_LINE_RE = re.compile(r"^\s*[A-Za-z][A-Za-z /]{1,30}:\s*(.*)$")
SECTION_KEYS = ("summary", "transcript", "action_items", "next_steps", "topics", "notes", "questions", "decisions")


class FirefliesParser(BaseMetadataParser):
    source_type = "fireflies"

    @staticmethod
    def _split_attendees(value: str) -> List[str]:
        names = []
        for part in re.split(r"[;,]|\band\b|[—–]|\s-\s", value or ""):
            name = display_name(part)
            if name and looks_like_full_name(name) and not name.lower().startswith("speaker"):
                names.append(name)
        return names

    def parse_header(self, content: str) -> Dict:
        header: Dict = {"attendees": [], "attendees_by_group": {}}
        m = MEETING_HEADER_RE.search(content or "")
        if not m:
            return header
        lines = content[m.end():].splitlines()
        current_group = None
        consumed = 0
        for line in lines[:60]:
            if not line.strip():
                if consumed:
                    break
                continue
            hm = HEADER_LINE_RE.match(line)
            if hm:
                consumed += 1
                key = hm.group(1).lower()
                group = hm.group(2) or hm.group(3)
                value = hm.group(4).strip()
                if key.startswith("attendees") or key.startswith("participants"):
                    current_group = (group or "all").strip()
                    names = self._split_attendees(value)
                    header["attendees"].extend(names)
                    header["attendees_by_group"].setdefault(current_group, []).extend(names)
                else:
                    current_group = None
                    if key in ("start time", "start"):
                        key = "time"
                    header[key] = value
                continue
            bm = BULLET_RE.match(line)
            if bm and current_group is not None:
                names = self._split_attendees(bm.group(1))
                header["attendees"].extend(names)
                header["attendees_by_group"][current_group].extend(names)
                continue
            if line.startswith("[") or line.lower().startswith("transcript"):
                break
            if OTHER_HEADER_LINE_RE.match(line):
                # unknown "Key: value" header line (Agenda, Recording link, ...): skip, keep reading
                current_group = None
                consumed += 1
                continue
            if consumed:
                break
        return header

    def _parse_metadata(self, md: DocumentMetadata, title: str, content: str) -> None:
        sections = split_pseudo_yaml(content)
        header = self.parse_header(content)

        date_value = header.get("date")
        created = None
        if date_value:
            # "2025-03-27 | Duration: 62 minutes" / "2026-01-14 15:00 UTC"
            created = normalize_date(date_value)
            created = combine_date_time(created, header.get("time"))
            if not header.get("duration") and "duration" in date_value.lower():
                dm = re.search(r"duration:\s*([^|]+)", date_value, re.IGNORECASE)
                if dm:
                    header["duration"] = dm.group(1).strip()
        md.created_at = created
        md.updated_at = created

        speakers = []
        for _, name in FIREFLIES_UTTERANCE_RE.findall(content or ""):
            name = display_name(name)
            if name and not is_bot(name) and len(name) <= 40 and not name.lower().startswith("speaker"):
                speakers.append(name)
        timestamps = FIREFLIES_UTTERANCE_RE.findall(content or "")

        md.participants = unique(header["attendees"] + speakers)
        md.authors = unique(header["attendees"][:1] or speakers[:1])
        topics = []
        if sections.get("topics"):
            topics = [t.strip("-• ").strip() for t in sections["topics"].splitlines() if t.strip()]
        md.extra.update({
            "meeting_title": header.get("title") or title,
            "duration": header.get("duration"),
            "location": header.get("location"),
            "attendees_by_group": {g: unique(n) for g, n in header["attendees_by_group"].items()},
            "speakers": unique(speakers),
            "topics": topics[:20],
            "sections": list(sections.keys()),
            "transcript_end": timestamps[-1][0] if timestamps else None,
        })
        groups = [g for g in header["attendees_by_group"] if g not in ("all",)]
        md.entities = unique(g for g in groups if g.lower() not in ("redwood", "as recorded"))

    def local_metadata(self, chunk_text: str, doc: Document, state: Dict) -> Dict:
        section = self.current_section(chunk_text, list(SECTION_KEYS), state)
        speakers = self.speakers_in_chunk(chunk_text, doc.metadata.extra.get("speakers", []))
        stamps = re.findall(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]", chunk_text)
        return {
            "section": section,
            "speakers": speakers,
            "ts_start": stamps[0] if stamps else None,
            "ts_end": stamps[-1] if stamps else None,
        }
