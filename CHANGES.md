# Change Record — 2026-09-03

**Metadata-aware query for EnterpriseRAG-Bench (phase 2: query side).** Structured query understanding, the hybrid score `α·dense + β·BM25 + γ·metadata + δ·level − λ·redundancy` with MMR selection, a corpus-wide hard filter, a metadata-in-text index variant, document-level MRR / nDCG, dev / test splits and a retrieval-only experiment runner. The legacy retriever path is byte-for-byte unchanged (verified against the saved smoke results and the real n800 results).

## 1. Query understanding (`src/query/understanding.py`, `src/query/time_expressions.py`, `src/prompt/rag_query.py`)

- `QueryConstraints`: intent, keywords, entities, people, projects, ticket keys, `time_range`, source types, channels. `parse_query_rules` (dates incl. quarters / halves / "early March 2026", ticket regex, project vocabulary + HubSpot accounts of the tree, hinted person names, explicit system names only), `normalize_llm_output` + `merge_constraints` (union, LLM window wins), `QueryUnderstanding` (mode `none` / `rules` / `llm`; sha1-keyed JSON cache `<save_dir>/query_cache/<model>_<prompt version>.json`, thread-safe). Trees without `documents` never call the model.
- Source systems are extracted only when named explicitly: content words (ticket, meeting, channel, pull request, deal, listing, ...) mislead on this benchmark (dev diagnostics, see §7).

## 2. Scoring and candidates (`src/query/scoring.py`, `src/query/candidates.py`, `src/query/credit.py`)

- `MetadataIndex` (per-node `NodeMetadataView`, source-type bits, ordinal-day intervals, ticket → nodes), `metadata_match` (weighted mean over specified fields; time = overlap coefficient), `level_score`, `TreeRelations` (ancestor / same-document), `select_with_mmr`, `hard_mask` with relaxation (`widen_time` → `drop_time` → `drop_source_type` → `drop_ticket_keys`).
- `DenseIndex` (normalised float32 matrix over all nodes, optional `.dense.npz` sidecar), `SparseScorer` (bm25s score vector, abstract = max over leaf descendants), `build_candidate_pool` keyed by node index.
- `merge_node_scores` (all layers in hybrid mode), `document_credit` (abstract nodes credit their documents only when `num_documents <= source_max_abstract_docs`; leaf-only variant).

## 3. Retriever (`src/tree_retriever.py`)

- `retrieve_mode="hybrid_score"` → `_retrieve_hybrid_score` (parse → dense sims + BM25 vector → pool with optional hard mask → metadata / level terms → MMR); `candidate_mode` `collapsed` / `traversal` (`_traverse` records every visited node; `_tree_retrieve` is a wrapper). `retrieve(..., extras=None, question_type=None)`; `layer_information` entries gain `sub_scores`; timing gains `parse` / `score`.
- Legacy path: `rrf_k` is now threaded into `rrf`; the runtime provenance header is skipped on metadata-in-text indexes.

## 4. Scripts and results

- `qa.py` / `main.py`: `question_type` and `extras` passed to the retriever (agentic sub-questions too); hybrid mode keeps nodes of every layer in `top_k_scores`; results gain `retrieved_doc_ids_leaf_only`, `query_parses` and (with `save_retrieval_diagnostics`) `retrieval_diagnostics`. `run_tag` (`result_stem`) names results / logs / judge caches.
- `experiments/make_splits.py` → `conf/enterprise_splits_s42.json` (dev 150 / test 350, stratified); `DataManager` honours `enterprise_split`.
- `experiments/run_arms.py`: arms A0, C0, C1, C1r, C2, C3, C3t, D, B, BC3 on one loaded tree (shared dense matrix, one query embedding per question), retrieval metrics + per-type table, `--grid` re-scores the C3 candidate pools for 108 weight combinations, `--weights-from`, `--save-results`.

## 5. Metadata-in-text index (arm B)

`enterprise_chunk_metadata_prefix=True`: `document_text_header(doc)` replaces the title prefix in `finalize_chunks`; `RAG._write_metadata_into_abstracts` prepends the aggregated header to every abstract and re-embeds it; tree / BM25 names get `_metatext` (`conf/enterprise_rag_metatext.py`, `conf/enterprise_rag_smoke_metatext.py`).

## 6. Evaluation (`src/evaluation.py`)

`docmrr` (`DocMRR`), `docndcg` (`DocNDCG@5/10`), `DocRecallLeaf@k` from `retrieved_doc_ids_leaf_only`; all with `_by_type` breakdowns. Config defaults for every new key live in `conf/__init__.py` (metadata-aware retrieval section).

## 7. Bug found on the way: stale `node.index` on layer-1 nodes

`AbstractTreeBuilder._construct_tree_with_preset_chunks` re-keyed the document-level (layer-1) nodes past the leaves but left `node.index` at the *document position*, so `node.index` of every layer-1 node pointed at an unrelated leaf (800 / 17 203 nodes in the n800 tree). Invisible in legacy mode (layer-1 nodes are never returned), but any path that returns abstract nodes credited the wrong document. Fixed at the source (`src/tree_builder/abstract.py`) and repaired in memory for existing pickles by `src.utils.repair_node_indices` (called in `RAG.load`, `RAG.add_documents`, `TreeRetriever.__init__`).

## 8. Field ablation (`experiments/run_arms.py --field-ablation`)

Re-scores the reference arm's candidate pools with `S_metadata` restricted to one field (and to all-but-one), writing `field_ablation.{json,md}`; answers "which metadata field helps / hurts" without new API calls.

## 9. Verification performed

- `pytest tests -q` (fast) and `-m slow` (offline e2e incl. hybrid_score and the metatext index) pass; new tests: `test_query_understanding.py`, `test_scoring.py`, `test_retriever_hybrid.py`, `test_splits.py`, additions to `test_evaluation.py` / `test_e2e_smoke.py`.
- Legacy regression: `qa.py --config enterprise_rag_smoke --run_tag smoke_legacy` reproduces the saved smoke results exactly; on the real n800 index 19 / 20 questions return identical `retrieved_doc_ids` (the 20th differs in the agentic second retrieval, whose sub-question is LLM-generated).
- n800 dev / test splits (retrieval only): §10 below, full tables under `output/experiments/n800_{dev,test}[_b]/` (`table.md`, `grid.json`, `field_ablation.md`).

## 10. Results on the n800 index (retrieval only, first retrieval, `rerank_top_k=10`)

`*` weights selected on dev by `--grid`: `{'alpha': 1.0, 'beta': 0.5, 'gamma': 0.25, 'delta': 0.0, 'lambda': 0.3}` (dev R@5 0.9301). Arms: A0 legacy traversal+RRF · C0 collapsed dense+BM25, no metadata · C1 / C1r + S_metadata (LLM / rules parse) · C2 + S_level · C3 + redundancy (full soft score) · C3t traversal candidates · D hard filter · B metadata-in-text index, legacy retrieval · BC3 metadata-in-text + full soft score.

**dev (150 questions)**

| arm | R@1 | R@5 | R@10 | MRR | nDCG@10 | extra@5 | R@5 constrained | R@5 project_related | R@5 completeness |
|---|---|---|---|---|---|---|---|---|---|
| A0 | 0.5186 | 0.8729 | 0.884 | 0.7647 | 0.7703 | 2.6809 | 1.0 | 0.5729 | 0.5329 |
| C0 | 0.7217 | 0.9168 | 0.9217 | 0.897 | 0.8782 | 1.9504 | 1.0 | 0.6042 | 0.5032 |
| C1 | 0.7198 | 0.8831 | 0.8831 | 0.8764 | 0.8477 | 1.8582 | 0.8889 | 0.5312 | 0.3569 |
| C1r | 0.7417 | 0.9012 | 0.902 | 0.9027 | 0.871 | 1.8227 | 0.8889 | 0.5602 | 0.3921 |
| C2 | 0.741 | 0.8625 | 0.8696 | 0.8791 | 0.8471 | 1.9433 | 0.8889 | 0.5312 | 0.3736 |
| C3 | 0.7417 | 0.8873 | 0.9014 | 0.8888 | 0.8692 | 2.773 | 0.9444 | 0.6968 | 0.3736 |
| C3t | 0.7347 | 0.8955 | 0.9109 | 0.8862 | 0.8706 | 2.6879 | 0.9444 | 0.6968 | 0.4014 |
| D | 0.7063 | 0.8352 | 0.8494 | 0.8516 | 0.8225 | 2.8652 | 0.8889 | 0.669 | 0.3736 |
| B | 0.4504 | 0.8785 | 0.8971 | 0.7289 | 0.7523 | 2.6809 | 1.0 | 0.6192 | 0.5722 |
| BC3 | 0.7347 | 0.9096 | 0.9397 | 0.8956 | 0.8878 | 3.0142 | 1.0 | 0.7211 | 0.4333 |

**test (350 questions)**

| arm | R@1 | R@5 | R@10 | MRR | nDCG@10 | extra@5 | R@5 constrained | R@5 project_related | R@5 completeness |
|---|---|---|---|---|---|---|---|---|---|
| A0 | 0.5486 | 0.8995 | 0.9106 | 0.7907 | 0.7989 | 2.5532 | 0.9524 | 0.6236 | 0.5342 |
| C0 | 0.7785 | 0.9058 | 0.9124 | 0.9277 | 0.8974 | 1.9574 | 0.9524 | 0.6524 | 0.5172 |
| C1 | 0.7813 | 0.9027 | 0.9111 | 0.9312 | 0.8966 | 1.8632 | 0.9524 | 0.5937 | 0.4557 |
| C1r | 0.7874 | 0.9025 | 0.91 | 0.9348 | 0.899 | 1.8632 | 0.9286 | 0.6048 | 0.4642 |
| C2 | 0.7813 | 0.9027 | 0.9111 | 0.9312 | 0.8966 | 1.8632 | 0.9524 | 0.5937 | 0.4557 |
| C3 | 0.781 | 0.9132 | 0.926 | 0.9274 | 0.9058 | 2.8693 | 0.9762 | 0.7181 | 0.4879 |
| C3t | 0.784 | 0.916 | 0.9324 | 0.9291 | 0.9098 | 2.8207 | 0.9762 | 0.7181 | 0.5189 |
| D | 0.726 | 0.8416 | 0.8543 | 0.8622 | 0.8391 | 2.8906 | 0.8333 | 0.6546 | 0.5408 |
| B | 0.5274 | 0.8905 | 0.906 | 0.7796 | 0.7918 | 2.4863 | 0.9524 | 0.6462 | 0.5272 |
| BC3 | 0.7664 | 0.9278 | 0.9437 | 0.9253 | 0.9114 | 2.9666 | 0.9762 | 0.7627 | 0.5271 |

**S_metadata field ablation on the test split** (C3 candidate pools, `S_metadata` restricted to one field; `n` = questions whose parse specifies that field)

| label | n_questions_with_field | DocRecall@1 | DocRecall@5 | DocRecall@10 | DocMRR | DocNDCG@5 | DocNDCG@10 |
|---|---|---|---|---|---|---|---|
| no_metadata | 0 | 0.7755 | 0.924 | 0.9374 | 0.9258 | 0.9112 | 0.9106 |
| all_fields | 185 | 0.781 | 0.9132 | 0.926 | 0.9274 | 0.9056 | 0.9058 |
| only_source_type | 54 | 0.7728 | 0.9227 | 0.9351 | 0.9251 | 0.9114 | 0.9097 |
| only_time | 39 | 0.7846 | 0.9245 | 0.9374 | 0.9316 | 0.9158 | 0.9149 |
| only_projects | 95 | 0.7778 | 0.9182 | 0.9311 | 0.9263 | 0.9074 | 0.9073 |
| only_entities | 63 | 0.7744 | 0.916 | 0.9274 | 0.9213 | 0.9022 | 0.9013 |
| only_ticket_keys | 4 | 0.7755 | 0.9262 | 0.9389 | 0.9258 | 0.9127 | 0.9117 |

What the numbers say (n800 = 722 gold documents + only 78 distractors, so constraints have little to separate):

- The collapsed candidate pool with node-index dedup (C0) is the main gain over the legacy traversal (test MRR 0.928 vs 0.791, R@1 0.78 vs 0.55); it needs no metadata at all.
- The soft metadata term does **not** raise overall recall (test R@5 C3 0.913 vs C0 0.906 comes from λ, not γ; dev is lower with γ>0). Per field, only `time` helps (MRR +0.006 on test); `source_type`, `entities`, `projects` hurt. Per question type, metadata helps `constrained` (0.976 vs 0.952) and `project_related` (0.718 vs 0.652) and hurts `completeness` (multi-document questions).
- The level term (δ) trades R@5 for R@1 / MRR (document-level nodes get credited); redundancy (λ=0.3) helps consistently.
- The hard filter (D) hurts everywhere (test R@5 0.842), even with source systems restricted to explicit names: questions describe the information, not the system, and document dates are often missing or broad. Relaxation never triggered (the pool stayed ≥ top_k).
- The metadata-in-text index (B) alone ≈ legacy; BC3 vs C3 flips sign between dev (0.910 vs 0.930) and test (0.928 vs 0.913) — noise-level on this subset.
**QA + LLM judge on the test split** (350 questions, agentic QA `max_retrieval_time=1`, gpt-4o-mini answers and judge; `experiments/qa_arms_n800.sh`, merged by `experiments/summarize_n800.py` → `output/experiments/n800_summary_test.md`). C3 / D use the default weights (γ 0.5, δ 0.3, λ 0.3); Cbest uses the dev-grid weights.

| arm | JudgeOverall | JudgeCorrectness | JudgeCompleteness | F1 | DocRecall@5 | DocMRR |
|---|---|---|---|---|---|---|
| A0 legacy | 0.3027 | 0.3600 | 0.3977 | 0.2681 | 0.9064 | 0.7970 |
| C0 collapsed hybrid, no metadata | **0.3202** | **0.3886** | **0.4023** | 0.2585 | 0.9134 | **0.9355** |
| C3 full soft score | 0.2911 | 0.3543 | 0.3791 | 0.2654 | 0.9021 | 0.9172 |
| Cbest dev-grid weights | 0.2698 | 0.3314 | 0.3707 | 0.2550 | **0.9156** | 0.9240 |
| D hard filter | 0.2827 | 0.3429 | 0.3653 | 0.2468 | 0.8495 | 0.8700 |
| B metatext index, legacy | 0.3077 | 0.3743 | 0.3994 | **0.2699** | 0.8938 | 0.7828 |
| BC3 metatext + full soft score | 0.2977 | 0.3600 | 0.3855 | 0.2598 | 0.8992 | 0.8999 |

JudgeOverall by question type (n: basic 123, semantic 87, intra_document_reasoning 28, project_related 28, constrained 21, completeness / conflicting_info / info_not_found / miscellaneous 14, high_level 7):

| arm | basic | completeness | conflicting_info | constrained | high_level | info_not_found | intra_doc | misc | project_related | semantic |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 0.458 | 0.000 | 0.129 | 0.084 | 0.000 | 0.071 | 0.628 | 0.464 | 0.014 | 0.237 |
| C0 | 0.472 | 0.107 | 0.119 | 0.080 | 0.000 | 0.143 | 0.773 | 0.554 | 0.024 | 0.197 |
| C3 | 0.401 | 0.036 | 0.119 | 0.152 | 0.000 | 0.071 | 0.660 | 0.607 | 0.000 | 0.221 |
| Cbest | 0.392 | 0.036 | 0.109 | 0.058 | 0.000 | 0.071 | 0.604 | 0.607 | 0.000 | 0.191 |
| D | 0.417 | 0.036 | 0.160 | 0.077 | 0.000 | 0.071 | 0.625 | 0.661 | 0.000 | 0.179 |
| B | 0.478 | 0.036 | 0.149 | 0.102 | 0.000 | 0.000 | 0.747 | 0.557 | 0.018 | 0.172 |
| BC3 | 0.414 | 0.036 | 0.109 | 0.066 | 0.000 | 0.071 | 0.676 | 0.632 | 0.018 | 0.236 |

- Only C0 improves answers as well as retrieval (JudgeOverall +0.018 over legacy, MRR +0.14). Every metadata arm (C3, Cbest, D, B, BC3) is at or below the legacy judge score even where its retrieval metrics are higher: the QA context is 10 nodes either way, and the metadata terms swap leaf chunks for document-level abstracts or metadata-matched chunks that the answer model uses less well.
- C3 is the only arm that lifts `constrained` questions (0.152 vs 0.084), matching the retrieval picture; it loses on `basic`. `high_level` scores 0 for every arm and `project_related` / `completeness` stay near 0: these need multi-document synthesis, not better single-document ranking.
- The first attempt at C3 / D / Cbest died on a transient `APIConnectionError` (tenacity `RetryError` propagates out of the threaded QA loop and aborts the whole run); a per-question retry / skip in `qa.py` would make long runs robust. The three arms were re-run with `experiments/qa_arms_n800_rerun.sh`.

## Files

Modified: `CHANGES.md`, `README.md`, `conf/__init__.py`, `conf/enterprise_rag.py`, `conf/enterprise_rag_smoke.py`, `eval.py`, `main.py`, `qa.py`, `src/dataset.py`, `src/evaluation.py`, `src/metadata/__init__.py`, `src/metadata/aggregate.py`, `src/model/factory.py`, `src/rag.py`, `src/tree_builder/abstract.py`, `src/tree_retriever.py`, `src/utils.py`, `tests/test_e2e_smoke.py`, `tests/test_evaluation.py`
New: `src/query/` (`understanding.py`, `time_expressions.py`, `scoring.py`, `candidates.py`, `credit.py`), `src/prompt/rag_query.py`, `experiments/` (`make_splits.py`, `run_arms.py`, `summarize_n800.py`, `qa_arms_n800.sh`), `conf/enterprise_rag_metatext.py`, `conf/enterprise_rag_smoke_metatext.py`, `conf/enterprise_splits_s42.json`, `tests/test_query_understanding.py`, `tests/test_scoring.py`, `tests/test_retriever_hybrid.py`, `tests/test_splits.py`
Generated (not committed): `output/experiments/n800_{dev,test}[_b]/`, `output/*.dense.npz`, `output/query_cache/`, the n800 `_metatext` tree + BM25 index.

## Next (not in this change)

- The 5k-document indexes (plain + `_metatext`, ~1 h of API each) where time / source constraints have real distractors to separate.
- Rethink `S_metadata` for multi-document (`completeness`) questions and entity-heavy parses (a per-type γ, or metadata as a re-ranking tie-breaker instead of an additive term).
- The 800 layer-1 nodes of existing pickles are repaired at load time only; rebuild to persist the fix.

# Change Record — 2026-09-02

**Metadata-aware ingestion for EnterpriseRAG-Bench (phase 1: ingestion side).** Rule-based per-source metadata parsers, a `Document` registry, document provenance on leaf and abstract nodes, a reproducible parquet subset loader, provenance in retrieval results, and document-level / LLM-judge evaluation. Query-side understanding and the hybrid score (α·dense + β·BM25 + γ·metadata + δ·level − λ·redundancy) are deliberately **not** part of this change; the aggregated metadata they need is stored now.

---

## 1. Three objects: Document, leaf node, abstract node

| Object | Where | Fields |
|---|---|---|
| `Document` — authoritative metadata | `src/metadata/document.py`; stored as `tree.documents[doc_id]` and `DataManager.document_registry` | `document_id`, `title`, `source_type`, `chunk_ids` (leaf indices), `metadata`: `authors`, `participants`, `channel`, `projects`, `entities`, `ticket_keys`, `emails`, `created_at`, `updated_at`, `time_range{start,end}`, `dates_mentioned`, `extra` (source-specific) |
| Leaf node (raw chunk) | `Node` (`src/utils.py`) | `document_id`, `local_metadata` (speakers / section / message sender of *this* chunk) — no copy of the document metadata |
| Abstract node (semantic cluster) | `Node` | `source_refs` (`[{document_id, chunk_ids}]`), `source_document_ids`, `aggregated_metadata` (`num_documents`, `num_chunks`, `source_types`, `authors`, `participants`, `projects`, `entities`, `ticket_keys`, `channels`, `time_range`, `latest_updated_at`, `source_authority`) |

- The document hierarchy and the semantic tree stay separate and are linked only through `document_id` / `source_refs`. Counts in `aggregated_metadata` are per unique document. Layer-1 nodes (one per document with the default `reorganize_leaf=False`) have a single `source_document_ids` entry; higher layers span several documents.
- "Authority / version": the dataset has no version field, so abstract nodes store `latest_updated_at` and a configurable per-source rank (`enterprise_source_authority`, confluence 5 … slack 1) as a heuristic for the later scoring work.
- `Node` keeps its positional constructor; the new fields are class-level defaults, so trees pickled before this change still load. `Tree(..., documents=None)` likewise.
- `src/metadata/aggregate.py::aggregate_tree_metadata(tree, registry)` fills `Document.chunk_ids` and every abstract node's fields in **one bottom-up post-pass** (memoised recursion over `children`, so it works for the exact, HNSW and bucketed builders without touching any node-creation call site). Called once from `RAG.add_documents`.
- Persistence: plain pickle is automatic; the bucketed chunk format (`src/tree_builder/chunks.py`) serialises the five new fields with `.get()` defaults and writes/reads `documents.json` next to the node shards (`bucketed-tree-v2`; v1 directories still load).

## 2. Per-source metadata parsers (`src/metadata/parsers/`, rules only, no LLM)

| Source | What is extracted |
|---|---|
| gmail | 95 % of `content` is a stringified Python/JSON list of messages with literal `\n`: `normalize_content` does `ast.literal_eval` → `json.loads` → `\n` unescape. Per message `From/To/Cc/Date/Subject/Attachments` (+ singular `Attachment`); authors = senders, participants = From ∪ To ∪ Cc, `created_at`/`updated_at` = first/last Date, `extra`: subject, num_messages, attachments, external_domains (also entities). Chunk `local_metadata`: `{message_index, from}`. |
| slack | `channel` = title when it matches `^[a-z][a-z0-9-]{1,30}$` (15 empty and ~11k junk titles → `None`, kept in `extra.raw_title`); speakers in the three line styles (`handle:`, `Name:`, `Name (role):`), bots (`*-bot`, `*Bot`) separated, roles kept; ``` fenced blocks are ignored so pasted `key: value` lines are not speakers. Dates only from prose. Chunk: `{speakers, channel}`. |
| fireflies | pseudo-YAML sections; `Meeting Header` block (`Date`, `Time`/`Start time`, `Duration`, `Location`, `Title`, `Agenda`, `Attendees`, `Attendees (Group):`, bulleted attendees) → `created_at` (date + time → UTC), `extra.attendees_by_group`; `[MM:SS] Name:` speakers; "Speaker N" placeholders dropped. Chunk: `{section, speakers, ts_start, ts_end}`. |
| jira / linear / github / hubspot | `split_pseudo_yaml` on `snake_case:` keys at column 0 (open vocabulary, synonyms folded: `resolution_notes→resolution`, `recent_activities/activity_timeline→timeline`, `review_conversation/review_thread→review_comments`, …); person slots (`reporter`, `assignee`, `reviewers`, …), `project` keys → `projects`, short status/priority/repo/… into `extra`, `created*/updated*/resolved*` keys → dates; **commenters** from comment-like sections in all observed shapes (`2026-03-10 Maya Chen:`, `2026-03-10 09:12 - Maya Patel (Support):`, `2025-05-07 (Liam O'Rourke, Support):`, `Aisha Kline (Support) 2026-03-10 02:20 —`, `Marco: … Priya (author): …`) with roles (`author`/`reporter` → authors). Jira: Title-Case sub-labels inside `description` (`Issue summary:`, `Impact:`, `Environment:`, `Customer:`). HubSpot: literal `\n` unescape, `title` = account (entity), `timeline` dates → created/updated. Chunk: `{section}`. |
| confluence / google_drive | `**Owners:**` / `Owner:` / `- Primary owner: Customer Success — Amira Patel` lines → authors (org phrases such as "Customer Success", "Eng SRE" are filtered), `Primary users` / `Stakeholders` / `Contacts` → participants, `**Slack channels:**` and inline `#channel` → `extra.slack_channels` + entities, markdown / Title-Case headings → `extra.sections`, `- Date: A to B` and `Last updated:` → dates. Chunk: `{section}` = nearest heading. |

Shared (`_finalize`): ticket keys `[A-Z]{2,6}-\d{1,6}` minus a stoplist (SHA, GPT, ISO, AWS, GPU, …) → `ticket_keys` and `entities`; emails; vocabulary projects/entities (`conf/enterprise_projects.json`, HubSpot account names of the subset are merged in automatically); `dates_mentioned` = header dates ∪ dates found in prose; `time_range` = min/max; `created_at`/`updated_at` fall back to the range bounds (`extra.created_at_source` = `header` | `inline`).

Dates (`src/metadata/dates.py`, stdlib only): RFC 2822, ISO 8601 (incl. `Z`, fractional seconds, `2026-01-14 15:00 UTC`), Gmail-UI `Tue, Jun 3, 2025 at 9:12 AM PT`, `March 27, 2025`, `27 March 2025`, `MM/DD/YYYY`, `date | Duration: …` suffixes; `PT/ET/CT/MT` resolved with US daylight-saving rules; output is date-only or UTC datetime so bounds sort lexicographically.

Parser coverage on the 800-document sanity subset (share of documents with a non-empty field): gmail authors/participants/created_at 100 %; fireflies 92 / 92 / 98 %; slack participants 100 %, channel 94 %; confluence created_at 57 %, projects 85 %; jira/linear created_at 90–99 %; github tickets 88 %. Authors stay empty for github/jira/linear/hubspot documents that have no reporter-style field.

## 3. Dataset loader and subset (`src/enterprise_rag.py`, `src/dataset.py`)

- `dataset="enterprise_rag"`: streams `data/enterpriseRAG-Bench/data/documents/test.parquet` (1.4 GB, one row group) with `pyarrow.parquet.ParquetFile.iter_batches`; never loads the content column at once.
- Subset = **all** documents referenced by any question's `expected_doc_ids` (722 unique) + `(enterprise_subset_size − 722)` distractors split evenly over the 9 source types, chosen by per-type reservoir sampling in one pass (`enterprise_subset_seed`). The 4 duplicate `doc_id`s are skipped (first occurrence kept). `enterprise_subset_size=None` keeps the full corpus. Cached as JSONL under `data/enterpriseRAG-Bench/subsets/enterprise_rag_n{size}_s{seed}.jsonl` with a meta header (rebuilt on mismatch).
- `DataManager`: `enterprise_kwargs=` (built by `enterprise_kwargs_from_conf(conf)`); `test_samples` truncates **questions only** (the corpus is controlled by the subset, so gold documents stay indexed); `gold_doc_ids`, `answer_facts`, `question_types`, `question_ids` from the questions parquet; `gold_answers` from `gold_answer`. `_preprocess_enterprise` parses every row with its source parser; `all_text_ids` holds document ids aligned with `all_passages`, so `document_index` keeps its positional meaning and the BM25 row == leaf index coupling is untouched.
- `finalize_chunks()` (called at the end of `split_dataset`) computes chunk `local_metadata` and, with `enterprise_chunk_title_prefix=True`, prefixes every chunk with `"{title}\n"` **after** chunking, so channel / account / ticket titles reach every leaf and the BM25 index while the semantic chunker sees clean sentences.
- `TreeBuilder.build_index(docs, use_multithreading, document_ids=None, chunk_metadata=None)` + `_attach_leaf_metadata()` give every leaf its `document_id` / `local_metadata` right after leaf creation (covers all four leaf-creation branches).
- Index / BM25 names use `dataset_tag(conf)` = `enterprise_rag_n{size}_s{seed}` so indexes of different subsets can never be paired.

## 4. Retrieval plumbing (no scoring change)

- `layer_information` entries gain `document_id`, `source_document_ids` (abstract nodes, ≤ 50), `source_type`, `title`, `local_metadata`.
- `context_metadata_header=True` prefixes every retrieved chunk with `[doc: <id> | <source_type> | <title> | <date> | <author or #channel>]` (abstract nodes: `[summary | N docs | <source types> | start..end]`), added after the text-keyed de-duplication so RRF / rerank behave as before.
- `qa.py` / `main.py`: `qa()` returns `sources` (unique documents of the retrieved leaves, best score first, via `collect_sources`); result files gain `sources` and `retrieved_doc_ids`; the log and chat mode print a `sources:` line.

## 5. Evaluation (`src/evaluation.py`, `src/prompt/rag_judge.py`)

- `docrecall`: `DocRecall@{1,2,5,10,20 ≤ top_k, all}` against `expected_doc_ids`; questions without gold documents (info_not_found, high_level) are skipped and counted (`DocRecall_n_evaluated` / `_n_skipped`); `DocRecall_by_type` breakdown.
- `extradocs`: `InvalidExtraDocs` / `@5` = retrieved documents that are not gold (lower is better).
- `llmjudge`: `judge_name` (e.g. `api:gpt-4o-mini`) grades correctness (binary) and completeness (share of `answer_facts` covered) from strict JSON; `JudgeOverall = mean(correct × completeness)`, `JudgeCorrectness`, `JudgeCompleteness`, `JudgeParseErrors`, `Judge_by_type`; verdicts cached per (question_id, answer hash) in `<save_dir>/results/<config>_judge.json`; parallel via `judge_workers`.
- `evaluate(..., retrieved_doc_ids=)`; metric failures are logged with a traceback instead of a silent `print(e)`.

## 6. Config, scripts, models

- New keys (`conf/__init__.py`, all CLI-overridable): `enterprise_data_dir`, `enterprise_subset_size=5000`, `enterprise_subset_seed=42`, `enterprise_subset_cache_dir`, `enterprise_project_vocab`, `enterprise_chunk_title_prefix=True`, `enterprise_source_authority`, `context_metadata_header=False`, `judge_name`, `judge_cache_dir`, `judge_model_kwargs`, `judge_workers=8`. `evaluation_metrics` accepts `docrecall`, `extradocs`, `llmjudge`.
- `conf/enterprise_rag.py` (API models, 5k subset, semantic chunking, `answer_type="medium"`, header on) and `conf/enterprise_rag_smoke.py` (offline: `hash:bow`, `fake:abstract`, `fake:qa`, 800 documents, 10 questions, `output/smoke`).
- `src/model/factory.py::build_model(name, task_type, conf)` replaces the three copies of `set_model` in `index.py` / `qa.py` / `main.py` and adds the offline backends `hash:` (signed bag-of-words hashing embeddings) and `fake:` (`src/model/fake.py`); `task_type="judge"` reuses the QA classes with `judge_*` kwargs.
- `README.md`: new "EnterpriseRAG-Bench with Metadata-Aware Ingestion" section.

### Fixes made along the way

- `src/tree_builder/abstract.py`: the layer-1 abstract loop used `range(0, max(keys), 10)` and skipped the last document whenever `(n_docs − 1) % 10 == 0`; now `max(keys) + 1` like the bucketed builder.

---

## Verification performed (`.venv/bin/python`, `mkdir -p log` first)

- `python -m pytest tests -q`: **65 passed** — dates (18 formats), every parser on real-format fixtures incl. the gmail stringified list and comment/commenter shapes, vocabulary, aggregation on a synthetic 2-document tree, bucketed + pickle round trips (old-format nodes still load), subset sampler on a synthetic parquet (gold kept, duplicates dropped, quotas, determinism, cache invalidation), doc-recall / extra-docs / judge parsing and caching, and an end-to-end offline build (`DataManager → split → RAG.add_documents → retrieve`) checking provenance headers and the BM25 coupling.
- Offline smoke: `index.py / qa.py / eval.py --config enterprise_rag_smoke` (800 documents, 13,441 leaves, 5 layers) → `DocRecall@10 = 1.0`, `InvalidExtraDocs = 4.6`.
- Real API sanity on the same 800-document subset with `conf/enterprise_rag.py` (`--enterprise_subset_size 800 --test_samples 20`): semantic chunking + `text-embedding-3-small`, `gpt-4o-mini` abstracts and QA, `gpt-4o-mini` judge. Tree: 16,184 leaves, 6 layers, cross-document nodes from layer 2 with populated `aggregated_metadata`. Results: `DocRecall@1/2/5 = 0.80/0.90/1.00`, `InvalidExtraDocs = 2.15`, `JudgeCorrectness = 0.40`, `JudgeCompleteness = 0.44`, `JudgeOverall = 0.36`, 0 parse errors. The full 5k build (`enterprise_subset_size=5000`) was **not** run.
- numpy prints `RuntimeWarning: divide by zero / overflow encountered in matmul` at `abstract.py:115` during tree building on macOS (Accelerate + float32 transposed views); all embeddings were verified finite and the trees are correct.

## Files

Modified: `README.md`, `conf/__init__.py`, `eval.py`, `index.py`, `main.py`, `qa.py`, `requirements.txt` (+ `pyarrow`, `pytest`), `src/dataset.py`, `src/evaluation.py`, `src/model/__init__.py`, `src/prompt/__init__.py`, `src/rag.py`, `src/tree_builder/abstract.py`, `src/tree_builder/base.py`, `src/tree_builder/chunks.py`, `src/tree_retriever.py`, `src/utils.py`
New: `src/metadata/` (`document.py`, `dates.py`, `patterns.py`, `sections.py`, `vocab.py`, `aggregate.py`, `parsers/{base,gmail,slack,fireflies,ticket,wiki}.py`), `src/enterprise_rag.py`, `src/model/factory.py`, `src/model/fake.py`, `src/prompt/rag_judge.py`, `conf/enterprise_rag.py`, `conf/enterprise_rag_smoke.py`, `conf/enterprise_projects.json`, `pytest.ini`, `tests/` (9 test modules + fixtures)

## Next (not in this change)

- Query understanding (intent, keywords, entities, people, projects, time and source constraints) and the metadata-aware hybrid score; retrieval de-duplication keyed by node index instead of chunk text; batched leaf embedding for API models; the 5k-document index run.

---

# Change Record — 2026-09-01

Two pieces of work, both uncommitted on `main`:

1. **Semantic chunking** replaces the static sentence-packing splitter as the default.
2. **OpenAI embeddings** (base URL + API key from `.env`) wired in as the default embedding backend.

Nothing has been committed; `git status` shows the modified/new files listed at the end.

---

## 1. Semantic chunking

### Behaviour

| Before | After |
|---|---|
| `split_text()` cut on `. ! ? \n`, dropped the delimiters, and packed sentences greedily up to `max_tokens_per_chunk`. | Every sentence is embedded; a new chunk starts where the distance between consecutive sentences spikes, still capped by `max_tokens_per_chunk`. |

Algorithm (`src/utils.py`):

- `split_sentences()` — keeps delimiters attached; `.`/`!`/`?` only end a sentence when followed by whitespace (so `3.14`, `www.example.com` survive); newlines always cut.
- `chunk_sentences_semantic()` — the reference `split_text_semantic` logic on pre-computed sentence embeddings:
  - split when `distance > threshold` **or** the chunk would exceed `max_tokens_per_chunk`;
  - threshold is the document's `semantic_threshold`-th **percentile** of consecutive-sentence distances by default (adapts to any embedding model); `semantic_threshold_type="absolute"` uses the raw distance like the reference;
  - single sentences longer than `max_tokens_per_chunk` go through the static `[,;:]` sub-splitter, and anything still over the bound (infobox-style runs with no punctuation) is hard-cut into token windows, so the bound always holds;
  - **short-chunk post-pass**: a chunk shorter than `short_chunk_tokens` is compared with its left and right neighbours (mean sentence embedding) and merged into the *more similar* one only if it is not semantically independent (distance ≤ `semantic_merge_threshold`, default = the document's split threshold) and the merged chunk still fits `max_tokens_per_chunk`. Independent short chunks are kept. No minimum-size guard is applied during splitting.
  - optional `distance_recorder` records per-chunk distances, split reason and merge history.
- `split_text_semantic()` — wrapper with the reference signature (`embedding_function`, …); falls back to static splitting if embedding fails.
- `split_text()` (static) is unchanged apart from accepting a tiktoken `Encoding` as well as an encoding name.

### Plumbing

- `src/model/embed.py` — `embed_batch(texts) -> (n, dim)` on `BaseEmbeddingModel` (per-item loop by default) with batched overrides for Ollama (`ollama.embed` returns all vectors; `embed()` still returns only the first), OpenAI, SentenceTransformers, vLLM and NV-Embed.
- `src/dataset.py` — `DataManager.split_text(chunking="static"|"semantic", …)`; the semantic path embeds all sentences of the corpus in one batched pass, then chunks each document and reassembles the original `List[str]` / `List[List[str]]` shape. New `split_dataset(data, conf)` replaces the four copy-pasted split blocks; it falls back to static splitting (with a log warning) when no embedding model is loaded (e.g. `eval.py`, `--no_retrieval`), where the chunks are never consumed anyway. With `tree_build_diagnostics=True` it prints chunking statistics and writes `<save_dir>/chunking_diagnostics_<dataset>.json`.
- `index.py`, `main.py`, `qa.py`, `eval.py` — the embedding model is now created **before** splitting; the split block is `split_dataset(data, conf)`.
- Save names — semantic indexes get a `_semantic` tag (`<…>_semantic_tree.pkl`, `bm25_<dataset>_semantic`) via `get_tree_save_name()` / new `get_sparse_save_name()` (`src/utils.py`, used by `src/tree_retriever.py`, `index.py`, `qa.py`). Static names are unchanged, so existing/HF-hosted static indexes still load and can never be paired with a semantic BM25 index.

### New config keys (`conf/__init__.py`, all CLI-overridable, e.g. `--chunking static`)

```python
chunking="semantic"                  # "semantic" | "static"
semantic_threshold=90                # percentile (0-100) or absolute distance, per semantic_threshold_type
semantic_threshold_type="percentile" # "percentile" | "absolute"
semantic_distance="cosine"           # "cosine" | "L1" | "L2" | "Linf"
short_chunk_tokens=30                # chunks shorter than this are checked for merging; 0 disables the post-pass
semantic_merge_threshold=None        # absolute distance for "not independent"; None = reuse the split threshold
semantic_embed_batch_size=64         # sentences per embed_batch call
```

### Fixes made along the way (outside the original scope, flagged)

- `src/tree_builder/base.py` — `_warm_up_embed_model()` loads a lazily-initialised local embedding model and runs one forward pass on the **main thread** before leaf nodes are embedded from the thread pool. Previously, concurrent `load_model()` calls from worker threads crashed sentence-transformers models (`Cannot copy out of meta tensor`) and, on macOS/MPS, the first forward pass inside a worker thread hangs forever. API-backed models are unaffected.
- `src/model/embed.py` — lazy `load_model()` is serialised with a lock.
- `index.py` / `eval.py` — unused `typing.List` import removed.

---

## 2. OpenAI embeddings via base URL + API key

- `src/model/embed.py` — `OpenAIEmbeddingModel` reads `OPENAI_API_KEY` and `OPENAI_BASE_URL` from the environment (empty/unset base URL → official `https://api.openai.com/v1`), accepts `dimensions`, embeds lists in one request, raises a clear error when the key is missing. Default model `text-embedding-3-small` (official ids carry no `openai/` prefix).
- `index.py`, `main.py`, `qa.py` — `"api": {"embed": OpenAIEmbeddingModel}` added to `set_model` (previously `api:` only existed for abstraction/QA).
- `conf/__init__.py` — default `embed_name` is now `api:text-embedding-3-small`; new dependency-free `load_env_file()` loads `<repo>/.env` on import (shell exports win; `KEY=` counts as unset).
- `conf/api_demo.py` (new) — embedding, abstraction and QA all via the API: `python index.py --config api_demo`.
- `.env.example` (new), `.env` added to `.gitignore`. The existing `.env` had a typo `OPEBAI_BASE_URL`; only the variable name was renamed to `OPENAI_BASE_URL`.
- `src/model/abstract.py`, `src/model/qa.py` — `api:` clients no longer KeyError on an unset `OPENAI_BASE_URL`, create `output/abs/` / `output/qa/` before writing logs, and use `rsplit('/')[-1]` so official model ids (no slash) don't crash.
- `README.md` — `.env` note in the prerequisites, semantic chunking keys/notes in the config listing and under *Step 3*, `api:` embedding example under *Changing LLM backbones*.

---

## Verification performed (`.venv/bin/python`, `mkdir -p log` needed first — pre-existing requirement)

- Unit checks with synthetic embeddings: sentence splitter, topic-boundary splits (percentile + absolute), token bound, long-sentence sub-split and hard cut, empty text, embedding-failure fallback, static byte-parity, merge post-pass (merge right / merge left / independent kept / skipped when over `max_tokens` / `short_chunk_tokens=0` = pure reference output / recorder), save names — all pass.
- `embed_batch` of `all-MiniLM-L6-v2` equals per-item `embed()`; base-class fallback checked.
- `index.py --config musique_summary --test_samples 20 --force_split true` with MiniLM: semantic → 58 chunks, max 100 tokens, tree connected, `_semantic` tree + BM25 + diagnostics JSON written; `--chunking static` → tree connected; `eval.py` runs (static fallback).
- Mock-based checks for `OpenAIEmbeddingModel` (request shape, out-of-order responses, `dimensions`, missing key, empty base URL) and `load_env_file()`.
- Real run with the user's key: `index.py --config api_demo --test_samples 5 --exclude_abs true` embedded 30 sentences + 12 chunks through `text-embedding-3-small` at api.openai.com; tree connected. No LLM calls were made.

## Files

Modified: `.gitignore`, `README.md`, `conf/__init__.py`, `eval.py`, `index.py`, `main.py`, `qa.py`, `src/dataset.py`, `src/model/abstract.py`, `src/model/embed.py`, `src/model/qa.py`, `src/tree_builder/base.py`, `src/tree_retriever.py`, `src/utils.py`
New: `.env.example`, `conf/api_demo.py`, `CHANGES.md`

## Known behaviour worth tuning

- With MiniLM on the musique sample, title-only chunks such as `"FC Barcelona"` stayed independent (their distance to the body exceeded the threshold); with `text-embedding-3-small` they merged. Raise `semantic_merge_threshold` or lower `semantic_threshold` if you want titles absorbed more aggressively.
- Switching between `chunking` modes never overwrites the other mode's index (different names); switching embedding models or thresholds within a mode still requires `--force_index_from_scratch --force_sparse_index_from_scratch`, as before.
