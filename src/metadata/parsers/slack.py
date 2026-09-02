"""Slack channel transcripts: "speaker: message" lines, title = channel name."""
import re
from typing import Dict, List

from ..document import Document, DocumentMetadata, unique
from ..patterns import CHANNEL_RE, SLACK_SPEAKER_LINE_RE, SPEAKER_STOPLIST, is_bot
from .base import BaseMetadataParser


CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


class SlackParser(BaseMetadataParser):
    source_type = "slack"

    @staticmethod
    def strip_code_fences(content: str) -> str:
        """Remove ``` fenced blocks (pasted logs / configs contain "key: value" lines that look like speakers)."""
        return CODE_FENCE_RE.sub("\n", content or "")

    @classmethod
    def extract_speakers(cls, content: str):
        speakers, bots, roles = [], [], {}
        counts = {}
        content = cls.strip_code_fences(content)
        for m in SLACK_SPEAKER_LINE_RE.finditer(content or ""):
            name, role = m.group(1).strip(), m.group(2)
            if name.lower() in SPEAKER_STOPLIST or len(name) > 40:
                continue
            if is_bot(name):
                bots.append(name)
                continue
            speakers.append(name)
            counts[name] = counts.get(name, 0) + 1
            if role and name not in roles:
                roles[name] = role.strip()
        return unique(speakers), unique(bots), roles

    def _parse_metadata(self, md: DocumentMetadata, title: str, content: str) -> None:
        channel = title.strip().lstrip("#")
        md.channel = channel if CHANNEL_RE.match(channel) else None
        speakers, bots, roles = self.extract_speakers(content)
        md.participants = speakers
        md.authors = speakers[:1]
        md.extra.update({
            "raw_title": title,
            "bots": bots,
            "roles": roles,
            "num_messages": len(SLACK_SPEAKER_LINE_RE.findall(content or "")),
        })
        if md.channel:
            md.entities = [f"#{md.channel}"]

    def local_metadata(self, chunk_text: str, doc: Document, state: Dict) -> Dict:
        candidates: List[str] = list(doc.metadata.participants) + list(doc.metadata.extra.get("bots", []))
        speakers = self.speakers_in_chunk(chunk_text, candidates)
        if speakers:
            state["last_speaker"] = speakers[-1]
        elif state.get("last_speaker"):
            speakers = [state["last_speaker"]]
        return {"speakers": speakers, "channel": doc.metadata.channel}
