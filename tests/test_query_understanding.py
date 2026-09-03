"""Query understanding: rules extractor, LLM JSON normalisation / merge, on-disk cache."""
import json
import os

from src.metadata.vocab import ProjectVocabulary
from src.prompt.rag_query import parse_query_response
from src.query import QueryConstraints, QueryUnderstanding, parse_query_rules, query_time_window
from src.query.understanding import merge_constraints, normalize_llm_output

VOCAB = ProjectVocabulary({"Apollo": ["Apollo", "project apollo"], "Hosted API": ["Hosted API"]},
                          {"Proxima Bank": ["Proxima Bank"]})


def test_time_windows():
    assert query_time_window("the Feb 12, 2026 incident") == ("2026-02-12", "2026-02-12")
    assert query_time_window("In Q4 2025, which segment") == ("2025-10-01", "2025-12-31")
    assert query_time_window("fourth quarter of 2025") == ("2025-10-01", "2025-12-31")
    assert query_time_window("early March 2026 upgrade") == ("2026-03-01", "2026-03-10")
    assert query_time_window("mid March 2026") == ("2026-03-11", "2026-03-20")
    assert query_time_window("late February 2026") == ("2026-02-21", "2026-02-28")
    assert query_time_window("during March 2026") == ("2026-03-01", "2026-03-31")
    assert query_time_window("H2 2025 roadmap") == ("2025-07-01", "2025-12-31")
    assert query_time_window("between 2026-01-10 and 2026-01-20") == ("2026-01-10", "2026-01-20")
    assert query_time_window("what shipped in 2025") == ("2025-01-01", "2025-12-31")
    assert query_time_window("Runtime 1.22 on Hosted pools") is None
    assert query_time_window("ticket INC-2026 last quarter") is None


def test_rules_parse():
    c = parse_query_rules("INC-9821: what did Maya Chen post in #eng-infra about Proxima Bank and project apollo "
                          "on 2026-01-15 in the Slack channel?", VOCAB)
    assert c.ticket_keys == ["INC-9821"]
    assert "INC-9821" in c.entities and "Proxima Bank" in c.entities
    assert c.projects == ["Apollo"]
    assert c.people == ["Maya Chen"]
    assert c.channels == ["eng-infra"]
    assert c.source_types == ["slack"]
    assert c.time_range == {"start": "2026-01-15", "end": "2026-01-15"}
    assert c.method == "rules" and c.has_constraints()
    assert set(c.specified_fields()) == {"source_type", "time", "people", "projects", "entities", "ticket_keys", "channels"}
    assert not parse_query_rules("What is the mission statement?", VOCAB).has_constraints()


def test_source_words_are_high_precision():
    assert parse_query_rules("which pull request changed the normalizer", VOCAB).source_types == []
    assert parse_query_rules("the GitHub PR that changed the normalizer", VOCAB).source_types == ["github"]
    assert parse_query_rules("the confluence runbook for rollback", VOCAB).source_types == ["confluence"]
    assert parse_query_rules("what page describes the SLA in the jira ticket", VOCAB).source_types == ["jira"]
    assert parse_query_rules("the incident ticket and the meeting notes in the channel", VOCAB).source_types == []
    assert parse_query_rules("the email thread with Acme about the Google Doc", VOCAB).source_types == ["gmail", "google_drive"]


def test_parse_query_response_tolerant():
    assert parse_query_response('```json\n{"intent": "lookup", "people": ["A B"]}\n```') == {"intent": "lookup", "people": ["A B"]}
    assert parse_query_response("Sure: {'x': 1}") is None
    assert parse_query_response("") is None and parse_query_response(None) is None
    assert parse_query_response('{"a": True, "b": None}') == {"a": True, "b": None}


def test_normalize_and_merge():
    data = {"intent": "Lookup", "keywords": ["429", "throttling"], "entities": ["proxima bank", "Streamly"],
            "people": ["Maya Chen (SRE)", None], "projects": ["Project Apollo"], "ticket_keys": ["inc-1", "INC-12"],
            "time_range": {"start": "March 3, 2026", "end": "2026-03-05"}, "source_types": ["Slack", "wiki"],
            "channels": ["#eng"]}
    llm = normalize_llm_output(data, VOCAB)
    assert llm.intent == "lookup" and llm.ticket_keys == ["INC-12"]
    assert llm.people == ["Maya Chen"] and llm.channels == ["eng"]
    assert llm.source_types == ["slack"]
    assert llm.time_range == {"start": "2026-03-03", "end": "2026-03-05"}
    assert llm.projects[0] == "Apollo" and llm.entities[0] == "Proxima Bank"
    rules = parse_query_rules("Proxima Bank 429 spike on 2026-03-04 in Q1 2026 (ADR-007)", VOCAB)
    merged = merge_constraints(llm, rules)
    assert merged.method == "llm+rules"
    assert merged.time_range == llm.time_range                      # LLM window wins
    assert "ADR-007" in merged.ticket_keys and "INC-12" in merged.ticket_keys
    assert merged.entities.count("Proxima Bank") == 1
    assert merge_constraints(None, rules) is rules
    assert QueryConstraints.from_dict(merged.to_dict()).to_dict() == merged.to_dict()


class StubModel:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def qa(self, messages, max_tokens=400, **kwargs):
        self.calls += 1
        return self.reply


def test_query_understanding_modes_and_cache(tmp_path):
    conf = {"query_understanding": "none", "save_dir": str(tmp_path)}
    assert not QueryUnderstanding(conf, VOCAB).parse("INC-1 on 2026-01-01").has_constraints()

    conf["query_understanding"] = "rules"
    assert QueryUnderstanding(conf, VOCAB).parse("INC-1 on 2026-01-01").method == "rules"

    model = StubModel('{"intent": "aggregation", "people": ["Bob Lim"], "time_range": null, "source_types": ["jira"]}')
    conf.update({"query_understanding": "llm", "query_name": "stub:model"})
    qu = QueryUnderstanding(conf, VOCAB, model=model)
    c1 = qu.parse("What did Bob Lim decide about INC-1 on 2026-01-01?")
    assert c1.method == "llm+rules" and c1.intent == "aggregation"
    assert c1.people == ["Bob Lim"] and c1.source_types == ["jira"] and c1.ticket_keys == ["INC-1"]
    assert c1.time_range == {"start": "2026-01-01", "end": "2026-01-01"}   # rules fallback window
    assert model.calls == 1
    qu.parse("What did Bob Lim decide about INC-1 on 2026-01-01?")
    assert model.calls == 1 and qu.stats["cache_hits"] == 1
    cache_file = os.path.join(str(tmp_path), "query_cache", "stub_model_v2.json")
    assert os.path.exists(cache_file) and len(json.load(open(cache_file))) == 1

    # a second instance reuses the cache; a broken reply degrades to rules
    qu2 = QueryUnderstanding(conf, VOCAB, model=StubModel("Thought: fake model.\nAnswer: nope"))
    assert qu2.parse("What did Bob Lim decide about INC-1 on 2026-01-01?").people == ["Bob Lim"]
    assert qu2.model.calls == 0
    c3 = qu2.parse("Something else about INC-2")
    assert c3.method == "rules" and c3.ticket_keys == ["INC-2"] and qu2.stats["parse_failures"] == 1
