"""Curated project / codename vocabulary (conf/enterprise_projects.json)."""
import json
import logging
import re
from typing import Dict, List, Optional, Tuple


class ProjectVocabulary:
    """
    {"projects": {"Redwood Private": ["Redwood Private", "private deploy"]},
     "entities": {"Triton": ["Triton"]}}
    `match(text)` returns the canonical names whose aliases occur in `text` (case-insensitive,
    word-bounded), in order of first appearance.
    """

    def __init__(self, projects: Optional[Dict[str, List[str]]] = None,
                 entities: Optional[Dict[str, List[str]]] = None) -> None:
        self.projects = self._normalise(projects)
        self.entities = self._normalise(entities)
        self._project_re, self._project_lookup = self._compile(self.projects)
        self._entity_re, self._entity_lookup = self._compile(self.entities)

    @staticmethod
    def _normalise(group: Optional[Dict[str, List[str]]]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for name, aliases in (group or {}).items():
            if isinstance(aliases, str):
                aliases = [aliases]
            aliases = [a for a in (aliases or []) if isinstance(a, str) and a.strip()]
            if name not in aliases:
                aliases.append(name)
            out[name] = aliases
        return out

    @staticmethod
    def _compile(group: Dict[str, List[str]]):
        lookup: Dict[str, str] = {}
        for name, aliases in group.items():
            for alias in aliases:
                lookup[alias.lower()] = name
        if not lookup:
            return None, lookup
        alternation = "|".join(re.escape(a) for a in sorted(lookup, key=len, reverse=True))
        return re.compile(rf"(?<![\w-])(?:{alternation})(?![\w-])", re.IGNORECASE), lookup

    @staticmethod
    def _find(pattern, lookup, text: str) -> List[str]:
        if pattern is None or not text:
            return []
        found = []
        seen = set()
        for m in pattern.finditer(text):
            name = lookup.get(m.group(0).lower())
            if name and name not in seen:
                seen.add(name)
                found.append(name)
        return found

    def match(self, text: str) -> Tuple[List[str], List[str]]:
        return (self._find(self._project_re, self._project_lookup, text),
                self._find(self._entity_re, self._entity_lookup, text))

    def is_empty(self) -> bool:
        return not self.projects and not self.entities

    @classmethod
    def load(cls, path: Optional[str]) -> "ProjectVocabulary":
        if not path:
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            logging.warning(f'Project vocabulary "{path}" not found; using an empty vocabulary.')
            return cls()
        return cls(data.get("projects"), data.get("entities"))
