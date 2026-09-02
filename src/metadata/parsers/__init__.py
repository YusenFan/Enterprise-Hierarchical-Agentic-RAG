from typing import Optional

from ..document import Document
from ..vocab import ProjectVocabulary
from .base import BaseMetadataParser
from .fireflies import FirefliesParser
from .gmail import GmailParser
from .slack import SlackParser
from .ticket import GithubParser, HubspotParser, JiraParser, LinearParser, PseudoYamlParser
from .wiki import ConfluenceParser, GoogleDriveParser, MarkdownishParser

ENTERPRISE_SOURCE_TYPES = (
    "confluence", "fireflies", "github", "gmail", "google_drive", "hubspot", "jira", "linear", "slack",
)

PARSERS = {
    "confluence": ConfluenceParser(),
    "fireflies": FirefliesParser(),
    "github": GithubParser(),
    "gmail": GmailParser(),
    "google_drive": GoogleDriveParser(),
    "hubspot": HubspotParser(),
    "jira": JiraParser(),
    "linear": LinearParser(),
    "slack": SlackParser(),
}


def get_parser(source_type: str) -> BaseMetadataParser:
    try:
        return PARSERS[(source_type or "").lower()]
    except KeyError:
        raise KeyError(
            f'No metadata parser for source_type "{source_type}". '
            f"Known source types: {sorted(PARSERS)}"
        ) from None


def parse_document(doc_id: str, source_type: str, title: str, content: str,
                   vocab: Optional[ProjectVocabulary] = None) -> Document:
    """Normalise + parse one raw corpus row. Returns the Document (use parser.normalize_content for the text)."""
    parser = get_parser(source_type)
    text = parser.normalize_content(content)
    return parser.parse(doc_id, source_type, title, text, vocab=vocab)
