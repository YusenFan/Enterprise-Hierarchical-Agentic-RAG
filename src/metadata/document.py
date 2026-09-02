"""
Document-level metadata objects.

A `Document` is the authoritative source of metadata for one corpus document. Leaf nodes
reference it through `document_id`; abstract nodes aggregate several documents through
`source_refs` (see src/metadata/aggregate.py). Documents never store the raw content: the
text lives in the dataset / chunks, which keeps the registry small enough to pickle with the tree.
"""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TimeRange:
    start: Optional[str] = None  # ISO 8601 ("2025-03-27" or "2025-03-27T15:00:00Z")
    end: Optional[str] = None

    def merge(self, other: Optional["TimeRange"]) -> "TimeRange":
        if other is None:
            return TimeRange(self.start, self.end)
        starts = [s for s in (self.start, other.start) if s]
        ends = [e for e in (self.end, other.end) if e]
        return TimeRange(min(starts) if starts else None, max(ends) if ends else None)

    def is_empty(self) -> bool:
        return self.start is None and self.end is None

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {"start": self.start, "end": self.end}


@dataclass
class DocumentMetadata:
    authors: List[str] = field(default_factory=list)        # sender / reporter / owners / first speaker
    participants: List[str] = field(default_factory=list)   # superset of authors: recipients, attendees, speakers
    channel: Optional[str] = None                           # slack channel name (validated), else None
    projects: List[str] = field(default_factory=list)       # vocabulary hits + explicit project fields
    entities: List[str] = field(default_factory=list)       # ticket keys, accounts, channels, vocabulary entities
    ticket_keys: List[str] = field(default_factory=list)
    emails: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    time_range: Optional[TimeRange] = None
    dates_mentioned: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)     # source-specific fields

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["time_range"] = self.time_range.to_dict() if self.time_range else None
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        data = dict(data or {})
        time_range = data.pop("time_range", None)
        md = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        if isinstance(time_range, dict):
            md.time_range = TimeRange(time_range.get("start"), time_range.get("end"))
        return md


@dataclass
class Document:
    document_id: str
    title: str
    source_type: str
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    chunk_ids: List[int] = field(default_factory=list)  # leaf node indices, filled by aggregate_tree_metadata
    num_chars: int = 0
    num_chunks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "source_type": self.source_type,
            "metadata": self.metadata.to_dict(),
            "chunk_ids": list(self.chunk_ids),
            "num_chars": self.num_chars,
            "num_chunks": self.num_chunks,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        return cls(
            document_id=data["document_id"],
            title=data.get("title", ""),
            source_type=data.get("source_type", ""),
            metadata=DocumentMetadata.from_dict(data.get("metadata", {})),
            chunk_ids=list(data.get("chunk_ids", [])),
            num_chars=int(data.get("num_chars", 0)),
            num_chunks=int(data.get("num_chunks", 0)),
        )


def unique(items, cap: Optional[int] = None) -> List:
    """Order-preserving de-duplication of hashable items (None / empty strings dropped)."""
    seen = set()
    out = []
    for item in items:
        if item is None or item == "":
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
        if cap is not None and len(out) >= cap:
            break
    return out
