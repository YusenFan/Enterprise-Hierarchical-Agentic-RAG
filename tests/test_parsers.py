import pytest

from src.metadata import ProjectVocabulary, get_parser, parse_document
from src.metadata.parsers import ENTERPRISE_SOURCE_TYPES, PARSERS
from src.metadata.patterns import extract_ticket_keys, find_full_names
from src.metadata.sections import bold_label_values, split_person_list, split_pseudo_yaml


def parse(fixture_text, name, source_type, title, vocab=None):
    parser = get_parser(source_type)
    text = parser.normalize_content(fixture_text(name))
    return parser, text, parser.parse("dsid_test", source_type, title, text, vocab=vocab)


def test_all_source_types_have_parsers():
    assert set(PARSERS) == set(ENTERPRISE_SOURCE_TYPES)
    with pytest.raises(KeyError):
        get_parser("notion")


def test_ticket_keys_stoplist():
    keys = extract_ticket_keys("see ENG-4129 and INC-2147, hash SHA-256, GPT-4o, us-east-1, RRB-17, ISO-8601, x ENG-4129 twice")
    assert keys == ["ENG-4129", "INC-2147", "RRB-17", "ENG-4129"]


def test_find_full_names_skips_org_phrases():
    names = find_full_names("Customer Success (Onboarding team) — Amira Patel; Eng SRE — Diego Morales")
    assert names == ["Amira Patel", "Diego Morales"]
    assert find_full_names("2026-03-10 (Liam O'Rourke, Support): x; Tomás Alvarez (HorizonPay SRE); GPU Burst Team") == \
        ["Liam O'Rourke", "Tomás Alvarez"]


def test_extract_commenters_formats():
    from src.metadata.parsers.ticket import extract_commenters
    cases = {
        "2026-03-10 Maya Chen: Logged on.\n2026-03-10 Liam O'Connor: Marking as P1.": (["Maya Chen", "Liam O'Connor"], {}),
        "2026-03-10 09:12 - Maya Patel (Support): Customer opened.": (["Maya Patel"], {"Maya Patel": "Support"}),
        "2025-05-19 10:14 PT — Jasmine Liu (Reporter)\nCustomer emailed Alice Wong.": (["Jasmine Liu"], {"Jasmine Liu": "Reporter"}),
        "2025-05-07 (Liam O'Rourke, Support): Created ticket.": (["Liam O'Rourke"], {}),
        "2026-03-10 09:12 - Support (Aisha Patel): Thanks.": (["Aisha Patel"], {}),
        "Aisha Kline (Support) 2026-03-10 02:20 — Logged alert. Escalated to Priya Shah.": (["Aisha Kline"], {"Aisha Kline": "Support"}),
        "2026-02-12 • Liam O'Connor: Initial.\n2026-02-25 • Priya Rao (Lead Eng): Engineering": (["Liam O'Connor", "Priya Rao"], {"Priya Rao": "Lead Eng"}),
        "Marco: Found a race. Priya: Added the mutex. Chen: Please gate.": (["Marco", "Priya", "Chen"], {}),
        "Diego Martin (review): Could we? -- requested change\nPriya Kapoor (author): Added.": (["Diego Martin", "Priya Kapoor"], {"Diego Martin": "review", "Priya Kapoor": "author"}),
        "Diego Morales (2026-02-20): Can we? Ava: made it configurable.": (["Diego Morales", "Ava"], {}),
    }
    for block, (names, roles) in cases.items():
        got_names, got_roles = extract_commenters(block)
        assert got_names == names, block
        assert got_roles == roles, block


def test_split_pseudo_yaml_keys_and_synonyms():
    sections = split_pseudo_yaml("release_notes:\nA\n\ndescription:\nB line\nImpact:\nx\n\nreview_thread: inline value\n")
    assert list(sections) == ["release_notes", "description", "review_comments"]
    assert sections["description"] == "B line\nImpact:\nx"
    assert sections["review_comments"] == "inline value"
    assert split_pseudo_yaml("just prose\nno keys here") == {}


# ---------------------------------------------------------------- gmail
def test_gmail_stringified_list(fixture_text):
    parser, text, doc = parse(fixture_text, "gmail_list.txt", "gmail", "Private upgrades: beta plan")
    md = doc.metadata
    assert text.startswith("From: Vivek Kulkarni")
    assert "\\n" not in text
    assert md.authors == ["Vivek Kulkarni", "Alyssa Chen"]
    assert {"Connor O'Brien", "Irene Choi", "Markus Klein"} <= set(md.participants)
    assert md.created_at == "2025-05-14T16:12:00Z"
    assert md.updated_at == "2025-05-15T15:03:00Z"
    assert md.extra["num_messages"] == 2
    assert md.extra["subject"] == "Private upgrades: beta plan w/ design partners"
    assert md.extra["attachments"] == ["Private_Upgrades_Beta_Plan_v0.3.pdf"]
    assert "cascadefg.com" in md.extra["external_domains"]
    assert "RRB-17" in md.ticket_keys
    assert "vivek_kulkarni@redwoodinference.com" in md.emails
    assert md.extra["created_at_source"] == "header"


def test_gmail_plain_and_singular_attachment(fixture_text):
    parser, text, doc = parse(fixture_text, "gmail_plain.txt", "gmail", "audit log requirements")
    md = doc.metadata
    assert md.authors == ["Irene Choi", "Kevin Osei"]
    assert md.created_at == "2025-06-10T09:12:00"
    assert md.updated_at == "2025-06-11T14:30:00Z"
    assert md.extra["attachments"] == ["redwood-private-upgrades-audit-events-v0.3.xlsx"]
    assert md.ticket_keys == ["ENG-4129"]
    # chunk-level metadata: message index / sender carry over across space-joined chunks
    state = {}
    chunk1 = " ".join(text.split("\n\n\n")[0].splitlines())
    chunk2 = " ".join(text.split("\n\n\n")[1].splitlines())
    assert parser.local_metadata(chunk1, doc, state) == {"message_index": 1, "from": "Irene Choi"}
    assert parser.local_metadata("continuation of the body", doc, state)["from"] == "Irene Choi"
    assert parser.local_metadata(chunk2, doc, state) == {"message_index": 2, "from": "Kevin Osei"}


# ---------------------------------------------------------------- slack
def test_slack_channel_speakers_bots(fixture_text):
    parser, text, doc = parse(fixture_text, "slack.txt", "slack", "eng-infra")
    md = doc.metadata
    assert md.channel == "eng-infra"
    assert md.authors == ["Sanaa"]
    assert md.participants == ["Sanaa", "Javier", "sasha", "Mei", "Priya", "Luca"]
    assert "routes" not in md.participants
    assert md.extra["bots"] == ["deploy-bot", "IncidentBot"]
    assert md.extra["roles"]["Sanaa"] == "infra PM"
    assert md.ticket_keys == ["INC-2147", "INC-2148"]
    assert "#eng-infra" in md.entities
    assert md.created_at == "2024-04-26T02:00:00Z"
    assert md.extra["created_at_source"] == "inline"
    chunk = " ".join(text.splitlines()[:4])
    assert parser.local_metadata(chunk, doc, {}) == {"speakers": ["Sanaa", "Javier"], "channel": "eng-infra"}


@pytest.mark.parametrize("title, channel", [("eng-runtime", "eng-runtime"), ("", None), ("1987654321", None),
                                            ("Weekly Sync Notes", None), ("#incidents", "incidents")])
def test_slack_channel_validation(title, channel):
    doc = get_parser("slack").parse("d", "slack", title, "alice: hi\nbob: hello")
    assert doc.metadata.channel == channel


# ---------------------------------------------------------------- fireflies
def test_fireflies_header_attendees_speakers(fixture_text):
    parser, text, doc = parse(fixture_text, "fireflies.txt", "fireflies", "AWS Marketplace listing review sync")
    md = doc.metadata
    assert md.created_at == "2026-01-14T18:03:00Z"        # 10:03 AM PT -> UTC
    assert md.extra["duration"] == "43 minutes"
    assert md.extra["attendees_by_group"] == {
        "Redwood": ["Jordan Blake", "Soojin Lee", "Chris Osei"],
        "AWS": ["Megan Li", "Andrew Patel"],
    }
    assert md.participants[:5] == ["Jordan Blake", "Soojin Lee", "Chris Osei", "Megan Li", "Andrew Patel"]
    assert "Speaker 2" not in md.participants and "Speaker 2" not in md.extra["speakers"]
    assert md.authors == ["Jordan Blake"]
    assert "AWS" in md.entities
    assert md.time_range.end == "2026-01-21"
    state = {}
    first = parser.local_metadata("summary: Introductory discovery between Redwood and Tethys", doc, state)
    assert first["section"] == "summary"
    chunk = "[00:00] Jordan Blake: Cool. Hey everyone. [00:10] Megan Li: Hi, Megan here."
    local = parser.local_metadata(chunk, doc, state)
    assert local == {"section": "summary", "speakers": ["Jordan Blake", "Megan Li"], "ts_start": "00:00", "ts_end": "00:10"}
    assert parser.local_metadata("transcript: Meeting Header Date: 2026-01-14", doc, state)["section"] == "transcript"


# ---------------------------------------------------------------- jira / linear / github / hubspot
def test_jira_sections_people_dates(fixture_text):
    parser, text, doc = parse(fixture_text, "jira.txt", "jira", "High P95 tail latency after quantization profile")
    md = doc.metadata
    assert md.authors == ["Dana Whitfield"]
    assert md.participants == ["Dana Whitfield", "Marcus Lin"]
    assert md.extra["sections"][:2] == ["description", "impact"] or "description" in md.extra["sections"]
    assert "resolution" in md.extra["sections"]            # resolution_notes -> resolution
    assert md.extra["description_sections"][:3] == ["Issue summary", "Impact", "Environment"]
    assert md.extra["customer"] == "LexiHealth"
    assert "LexiHealth" in md.entities
    assert set(md.ticket_keys) == {"SUP-2211", "ENG-4187"}
    assert md.created_at == "2026-03-13T13:22:00Z"
    assert md.updated_at == "2026-03-14"
    state = {}
    assert parser.local_metadata("description: Issue summary: Customer deployed", doc, state)["section"] == "description"
    assert parser.local_metadata("more prose without a key", doc, state)["section"] == "description"
    assert parser.local_metadata("logs: 2026-03-13T14:12:03Z request_id=02c9", doc, state)["section"] == "logs"


def test_linear_project_and_comment_dates(fixture_text):
    parser, text, doc = parse(fixture_text, "linear.txt", "linear", "Accessible API Onboarding")
    md = doc.metadata
    assert md.projects == ["Embeddings & Rerank quickstart"]
    assert set(md.ticket_keys) == {"ENG-4210", "PM-331"}
    assert md.time_range.start == "2026-03-10" and md.time_range.end == "2026-03-12"


def test_github_reviewers_and_stoplist(fixture_text):
    parser, text, doc = parse(fixture_text, "github.txt", "github", "introduce-pluggable-attn-adapter")
    md = doc.metadata
    assert md.extra["reviewers"] == ["Jordan Kim", "Priya Nair", "Luis Martinez"]
    assert md.participants == ["Jordan Kim", "Priya Nair", "Luis Martinez"]
    assert "Priya" not in md.participants
    assert set(md.ticket_keys) == {"ENG-4129", "ENG-4187"}     # SHA-256 / GPT-4o excluded
    assert md.extra["sections"] == ["release_notes", "description", "review_comments"]


def test_hubspot_account_timeline_unescape(fixture_text):
    parser, text, doc = parse(fixture_text, "hubspot.txt", "hubspot", "Orion Medical Imaging AI")
    md = doc.metadata
    assert "\\n" not in text and "Account background:\n- Mid-sized" in text
    assert md.extra["account"] == "Orion Medical Imaging AI"
    assert md.entities[0] == "Orion Medical Imaging AI"
    assert md.created_at == "2026-03-02" and md.updated_at == "2026-03-12"   # last timeline entry
    assert md.time_range.end == "2026-03-16"                                  # target date mentioned inline
    assert md.extra["timeline_entries"] == 3
    assert "DEAL-771" in md.ticket_keys


# ---------------------------------------------------------------- confluence / google_drive
def test_confluence_bold_labels_owners_channels(fixture_text):
    parser, text, doc = parse(fixture_text, "confluence.txt", "confluence", "Runbook: perf-canary")
    md = doc.metadata
    assert md.authors == ["Vanessa Ortiz", "Noah Patel", "Sean Gallagher", "Rafael Mendes", "Amira Patel", "Diego Morales"]
    assert md.extra["slack_channels"] == ["eng-runtime", "eng-oncall", "sre-oncall"]
    assert "#eng-runtime" in md.entities
    assert md.extra["service_name"] == "perf-canary"
    assert "Quick reference" in md.extra["sections"]
    assert md.updated_at == "2026-02-03"
    state = {}
    assert parser.local_metadata("## Purpose This runbook describes", doc, state)["section"] == "Purpose"
    assert parser.local_metadata("continuation without heading", doc, state)["section"] == "Purpose"


def test_google_drive_date_range_and_owner(fixture_text):
    parser, text, doc = parse(fixture_text, "google_drive.txt", "google_drive", "Dispatch surface playback memo")
    md = doc.metadata
    assert md.created_at == "2026-05-15" and md.updated_at == "2026-05-17"
    assert md.authors == ["Priya Natarajan"]
    assert "Experiment setup (short)" in md.extra["sections"]


# ---------------------------------------------------------------- vocabulary + parse_document
def test_vocab_and_parse_document(fixture_text):
    vocab = ProjectVocabulary(projects={"Redwood Private": ["Private upgrade", "Redwood Private"]},
                              entities={"Cascade Financial Group": ["Cascade Financial"]})
    doc = parse_document("dsid_1", "gmail", "beta plan", fixture_text("gmail_list.txt"), vocab=vocab)
    assert doc.metadata.projects == ["Redwood Private"]
    assert "Cascade Financial Group" in doc.metadata.entities
    assert doc.document_id == "dsid_1" and doc.source_type == "gmail" and doc.num_chars > 100
    roundtrip = type(doc).from_dict(doc.to_dict())
    assert roundtrip.metadata.created_at == doc.metadata.created_at
    assert roundtrip.metadata.time_range.start == doc.metadata.time_range.start


def test_split_person_list_and_bold_labels():
    assert split_person_list("Infra: Vanessa Ortiz; SRE: Sean Gallagher / Rafael Mendes") == \
        ["Vanessa Ortiz", "Sean Gallagher", "Rafael Mendes"]
    labels = bold_label_values("**Owners:**\n- Infra: Vanessa Ortiz\n\nOwner: Noah Patel\n")
    assert labels == {"owners": "Infra: Vanessa Ortiz", "owner": "Noah Patel"}
