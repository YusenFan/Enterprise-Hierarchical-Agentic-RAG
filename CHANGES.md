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
