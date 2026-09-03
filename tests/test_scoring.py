"""Hybrid-score terms: metadata views / matching, level preference, redundancy, MMR, hard mask."""
import numpy as np

from src.metadata import aggregate_tree_metadata
from src.query import QueryConstraints
from src.query.scoring import (Candidate, MetadataIndex, TreeRelations, level_score, metadata_match,
                               normalize_scores, select_with_mmr, time_overlap)
from src.utils import reverse_mapping
from tests.test_aggregate import build_tree


def indexed_tree():
    tree, registry = build_tree()
    registry["doc_b"].metadata.ticket_keys = ["ENG-1"]
    aggregate_tree_metadata(tree, registry)
    tree.all_nodes[0].local_metadata = {"section": "s0", "speakers": ["Carol Wu"]}
    node_to_layer = reverse_mapping(tree.layer_to_node_indices)
    return tree, MetadataIndex(tree, node_to_layer)


def test_views_leaf_and_abstract():
    tree, index = indexed_tree()
    leaf = index.view(0)
    assert leaf.is_leaf and leaf.document_ids == ["doc_a"] and leaf.source_types == {"slack"}
    assert {"alice tan", "carol wu"} <= leaf.people and leaf.channels == {"eng"}
    assert leaf.projects == {"apollo"} and leaf.t_start is not None and leaf.t_end - leaf.t_start == 1
    root = index.view(6)
    assert not root.is_leaf and root.num_documents == 2 and root.source_types == {"jira", "slack"}
    assert {"alice tan", "bob lim"} <= root.people and root.ticket_keys == {"ENG-1"}
    assert index.view(4).t_start == leaf.t_start and index.view(5).ticket_keys == {"ENG-1"}
    assert index.src_bits[0] != 0 and index.t_start[6] == leaf.t_start


def test_metadata_match_fields():
    tree, index = indexed_tree()
    c = QueryConstraints(source_types=["jira"], people=["A. Tan", "Bob"], projects=["Apollo"],
                         ticket_keys=["ENG-1"], time_range={"start": "2026-08-13", "end": "2026-08-14"},
                         channels=["eng"])
    score_b, fields_b = metadata_match(c, index.view(2), tol_days=0)          # doc_b leaf
    assert fields_b["source_type"] == 1.0 and fields_b["people"] == 1.0 and fields_b["projects"] == 1.0
    assert fields_b["ticket_keys"] == 1.0 and fields_b["time"] == 1.0 and fields_b["channels"] == 0.0
    score_a, fields_a = metadata_match(c, index.view(0), tol_days=0)          # doc_a leaf
    assert metadata_match(c, index.view(0))[1]["time"] == 1.0                # default 7-day tolerance
    assert fields_a["source_type"] == 0.0 and fields_a["channels"] == 1.0 and fields_a["time"] == 0.0
    assert fields_a["people"] == 0.5 and fields_a["ticket_keys"] == 0.0
    assert score_b > score_a
    assert metadata_match(QueryConstraints(), index.view(0)) == (0.0, {})
    # unknown dates score 0 on time only, the other fields still count
    _, fields_unknown = metadata_match(QueryConstraints(time_range={"start": "2026-08-13", "end": "2026-08-14"},
                                                        projects=["Apollo"]), index.view(6))
    assert fields_unknown["projects"] == 1.0 and fields_unknown["time"] == 1.0     # root spans both docs


def test_time_overlap_coefficient():
    d = lambda s: __import__("datetime").date.fromisoformat(s).toordinal()
    assert time_overlap(d("2025-10-01"), d("2025-12-31"), d("2025-11-03"), d("2025-11-03")) == 1.0
    assert time_overlap(d("2025-10-01"), d("2025-12-31"), d("2026-01-05"), d("2026-01-05")) == 0.0
    assert time_overlap(d("2025-10-01"), d("2025-12-31"), d("2026-01-05"), d("2026-01-05"), tol_days=7) == 1.0
    assert time_overlap(d("2026-03-01"), d("2026-03-10"), d("2026-03-06"), d("2026-03-20")) == 0.5
    assert time_overlap(d("2026-03-01"), d("2026-03-10"), None, None) == 0.0


def test_level_score():
    assert level_score("basic", 0) == 1.0 and level_score("basic", 1) == 0.5 and level_score("basic", 3) == 0.0
    assert level_score("high_level", 2) == 1.0 and level_score("high_level", 0) == 0.2
    assert level_score(None, 0) == 1.0                              # default table
    assert level_score("basic", 0, {"basic": {"0": 0.1, "1": 0.2, "2+": 0.3}}) == 0.1
    assert level_score("weird", 1, {"x": {"0": 1.0}}) == 0.5        # nothing applies


def test_relations_and_mmr():
    tree, index = indexed_tree()
    rel = TreeRelations(tree, index.views)
    assert rel.related(0, 4) == 1.0 and rel.related(6, 2) == 1.0 and rel.related(0, 0) == 1.0
    assert rel.related(0, 1) == 0.5 and rel.related(0, 2) == 0.0 and rel.related(4, 5) == 0.0
    cands = [Candidate(0, 0, dense=0.9), Candidate(1, 0, dense=0.85), Candidate(2, 0, dense=0.8),
             Candidate(4, 1, dense=0.7, level=1.0)]
    plain = select_with_mmr(cands, {"alpha": 1, "beta": 0, "gamma": 0, "delta": 0, "lambda": 0}, 3, rel)
    assert [c.node_index for c, _, _ in plain] == [0, 1, 2]
    mmr = select_with_mmr(cands, {"alpha": 1, "beta": 0, "gamma": 0, "delta": 0, "lambda": 0.6}, 3, rel)
    assert [c.node_index for c, _, _ in mmr] == [0, 2, 1]           # same-doc leaf pushed back
    assert mmr[1][2]["redundancy"] == 0.0 and mmr[2][2]["redundancy"] == 0.5
    lvl = select_with_mmr(cands, {"alpha": 1, "beta": 0, "gamma": 0, "delta": 1.5, "lambda": 0}, 1, rel)
    assert lvl[0][0].node_index == 4 and lvl[0][2]["level"] == 1.0
    assert normalize_scores([1, 1, 1]).tolist() == [0.5, 0.5, 0.5]
    assert normalize_scores([3, 1, 2], "rank").tolist() == [1.0, 0.0, 0.5]


def test_hard_mask_and_unknown_pass():
    tree, index = indexed_tree()
    fields = ["ticket_keys", "time", "source_type"]
    mask, applied = index.hard_mask(QueryConstraints(source_types=["jira"]), fields, 0)
    assert applied == ["source_type"]
    assert mask[2] and mask[3] and mask[5] and mask[6] and not mask[0] and not mask[4]
    mask, applied = index.hard_mask(QueryConstraints(time_range={"start": "2026-08-15", "end": "2026-08-16"}), fields, 0)
    assert applied == ["time"] and not mask[0] and mask[2] and mask[6]
    mask, _ = index.hard_mask(QueryConstraints(time_range={"start": "2026-08-15", "end": "2026-08-16"}), fields, 5)
    assert mask[0]                                                  # tolerance widens the window
    mask, applied = index.hard_mask(QueryConstraints(ticket_keys=["ENG-1"]), fields, 0)
    assert applied == ["ticket_keys"] and mask[0] and mask[2] and mask[6]   # doc_a has no key -> unknown -> passes
    index.ticket_nodes.setdefault("OPS-9", set()).update({0, 1, 4})
    mask, _ = index.hard_mask(QueryConstraints(ticket_keys=["ENG-1"]), fields, 0)
    assert not mask[0] and mask[2]                                  # now doc_a mentions another key -> fails
    mask, applied = index.hard_mask(QueryConstraints(ticket_keys=["ZZZ-1"]), fields, 0)
    assert applied == [] and mask.all()                             # unknown key: nothing to filter on
    mask, applied = index.hard_mask(QueryConstraints(people=["Alice Tan"]), fields, 0)
    assert applied == [] and mask.all()
