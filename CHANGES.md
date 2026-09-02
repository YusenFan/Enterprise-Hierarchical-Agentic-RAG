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
