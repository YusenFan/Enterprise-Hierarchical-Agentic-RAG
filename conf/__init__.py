import os
import ast
import importlib
from typing import Sequence, Dict, TypedDict
from pathlib import Path


def load_env_file(path: str | Path | None = None, override: bool = False) -> Dict[str, str]:
    """
    Load KEY=VALUE pairs from a ".env" file into os.environ (no extra dependency).
    Defaults to "<repo root>/.env". Variables already exported in the shell win
    unless override=True. Used for OPENAI_BASE_URL / OPENAI_API_KEY of the "api:" backends.
    """
    if path is None:
        path = Path(__file__).resolve().parent.parent / ".env"
    path = Path(path)
    loaded: Dict[str, str] = {}
    if not path.is_file():
        return loaded
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, value = line.split("=", maxsplit=1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", maxsplit=1)[0].rstrip()
        if not key or value == "":   # "KEY=" means unset
            continue
        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = value
    return loaded


load_env_file()


class Config(TypedDict, total=False):
    # ===================================== Data config =====================================
    # Dataset name: "nq", "popqa", "hotpotqa", "2wikimultihopqa", "musique", 
    #   "multihoprag", "narrativeqa", "infinitybench_longbook", "qmsum", "wcep"
    #   If read_local_pdf is not None, dataset name will be automatically set.
    dataset: str
    # Dataset directory, default to "data/".
    data_dir: Path
    # Read and parse local PDF(s) as a local dataset. MinerU is needed in your environment.
    #   This will automatically reset the dataset name. 
    #   Default to None to use predefined dataset names.
    #   Set to "file" to read a single PDF file. data_dir must be a single ".pdf" file.
    #   Set to "dir" to read a directory of PDFs. data_dir must be a dir path.
    #   Set to "dir_recursive" to read a directory and its subdirectories recursively.
    #   Set to "package" to read a compressed package of PDFs. data_dir must be an archive file 
    #   (.zip, .tar, .tar.gz, .rar, .7z, etc).
    #   Set to "package_recursive" to read a compressed package and search recursively.
    #   Note that evaluation scripts (main.py & eval.py) do not support local PDFs.
    read_local_pdf: str | None
    # Batch size for processing local PDFs with MinerU. Only used when read_local_pdf is not None. 
    pdf_batch_size: int
    # Number of samples for testing with part of the dataset. Data becomes data[:test_samples].
    test_samples: int
    # Always split the data even there are preset chunks. Used when preset chunks are 
    #   long passages, e.g., multihoprag, infinitybench-longbook, qmsum, wcep.
    force_split: bool
    # Tokenizer name (from tiktoken) for spliting datasets with no preset chunks.
    tokenizer: str
    # Maximum token count of one chunk when tokenizer splits the data.
    max_tokens_per_chunk: int
    # Chunking method used when the data is split, ("semantic", "static").
    #   "static" packs sentences greedily up to max_tokens_per_chunk.
    #   "semantic" embeds every sentence with the embedding model and starts a new chunk
    #   where the distance between consecutive sentences spikes (still capped by
    #   max_tokens_per_chunk). Requires the embedding model, so it is only effective
    #   where the index is built (index.py / main.py); other scripts fall back to "static".
    #   Semantic indexes are saved with a "_semantic" tag (tree file and bm25 directory).
    chunking: str
    # Split threshold for semantic chunking, interpreted according to semantic_threshold_type.
    semantic_threshold: float
    # ("percentile", "absolute"). "percentile": split where a sentence distance exceeds the
    #   semantic_threshold-th percentile of the document's consecutive-sentence distances
    #   (adapts to any embedding model). "absolute": split where distance > semantic_threshold.
    semantic_threshold_type: str
    # Metric for sentence / chunk distances in semantic chunking, ("cosine", "L1", "L2", "Linf").
    semantic_distance: str
    # Semantic chunks shorter than this many tokens are checked for merging: a short chunk is
    #   merged into its more similar neighbour (left or right) only if it is not semantically
    #   independent, i.e. that distance <= semantic_merge_threshold, and the merged chunk still
    #   fits max_tokens_per_chunk. Independent short chunks are kept. Set to 0 to disable.
    short_chunk_tokens: int
    # Absolute distance below which a short chunk counts as "not independent" from its neighbour.
    #   Default to None to reuse the document's effective split threshold.
    semantic_merge_threshold: float | None
    # Number of sentences per embedding call during semantic chunking.
    semantic_embed_batch_size: int

    # =================================== Embedding config ===================================
    # Model name for embedding documents and queries, "[PLATFORM]:[MODEL_NAME_OR_PATH]"
    #   e.g., "api:text-embedding-3-small" (OpenAI-compatible endpoint; set OPENAI_API_KEY and 
    #   optionally OPENAI_BASE_URL in ".env"), "transformers:nvidia/NV-Embed-v2", 
    #   "transformers:facebook/contriever", "sentence-transformers:multi-qa-mpnet-base-cos-v1", 
    #   "ollama:qwen3-embedding"
    embed_name: str
    # Cache directory of the embedding model, default to os.environ["HF_HOME"].
    embed_cache_dir: Path
    # Model kwargs for embedding model. Keys depend on the model you use. e.g.: 
    #   General model args: "temperature" "top_k" "top_p" "stop"
    #   Transformers-specific args: "frequency_penalty" "presence_penalty" "repetition_penalty"
    #   vLLm-specific args: "tensor_parallel_size" "gpu_memory_utilization" "max_model_len"
    #   Ollama-specific model args: "num_ctx" "num_predict" "repeat_penalty" "dimensions"
    embed_model_kwargs: Dict

    # ================================= Tree indexing config =================================
    # Tree builder backend. Default to "exact", i.e., the original dense similarity ranking.
    #   "hnsw" replaces the similarity ranking with HNSW candidate edges.
    tree_builder: str
    # The single-document retrieval setup, i.e., an independent tree index for each document.
    #   Used for passage-level and document-level datasets whose queries are passage-specific, 
    #   e.g., narrativeqa, infinitybench_longbook, qmsum, wcep.
    passage_as_tree: bool
    # Reorganize leaves such that chunks from different documents could share a parent node. 
    #   Default to False and chunks from the same document will be automatically merged. 
    #   Set this on when there are preset chunks to be manually split (e.g., multihoprag).
    #   Forced True when no preset chunks and force_split=False. 
    reorganize_leaf: bool
    # Maximum number of children per abstract node. Should be >3. 
    #   If a node has more than max_num_children children, tree rebalancing will be activated 
    #   to split the node and its children into multiple subtrees.
    #   You would like to set this value not smaller than 2 * tree_top_k 
    #   to ensure the amount of retrieved documents.
    #   Note: if the data has preset chunks and force_split=False, 
    #   the last abstract layer (penaltimate layer) will NOT be checked nor reorganized.
    max_num_children: int | None
    # Extra tree building diagnostics. Includes: token stats, indexing time & memory, 
    #   connectivity checks, etc. Default to False.
    tree_build_diagnostics: bool
    # ===== Bucketing config =====
    # Maximum bucket size for the bucketed tree builder. Default to None (bucketing disabled).
    #   Bucketing first partition a large corpus into multiple buckets using a fast 
    #   recursive spherical $k$-means. Bucket trees are merged at the highest layer. 
    #   Use this and the params below when your corpus is fairly large.
    bucket_size: int | None
    # How many sub-buckets can a single bucket be further divided into during recursive bucketing.
    #   The real fanout value is automatically calculated; this sets the upper bound.
    bucket_max_fanout: int
    # Maximum sample count used by recursive bucketing. Useful if the bucket size is very large
    #   which leads to inefficient k-means. 
    bucket_sample_size: int
    # Number of coarse k-means refinement steps used by recursive bucketing.
    bucket_kmeans_iters: int
    # Chunk size for saving bucketed trees ONLY. The entire tree index will be saved 
    #   in multiple files with each file containing at most tree_save_chunk_size nodes.
    #   Set to None to save in a single file. 
    tree_save_chunk_size: int | None
    # Partition ratio for original similarity ranking. Useful when the corpus or bucket 
    #   is fairly large. Set this >1 for faster-but-not-as-accurate similarity ranking.
    partition_ratio: float
    # ===== HNSW config ===== 
    # Candidate neighbor count for HNSW tree building. Only used when tree_builder is "hnsw".
    hnsw_top_k: int
    # The number of bi-directional links created for each new node during HNSW construction. 
    #   Higher value leads to better connectivity but slower construction and more memory usage.
    hnsw_m: int
    # The search width during HNSW construction. Higher value leads to better graph 
    #   but slower construction.
    hnsw_ef_construction: int
    # The search width during HNSW query. Higher value leads to better recall but slower speed.
    hnsw_ef_search: int

    # =================================== Abstraction config ==================================
    # Do not generate abstracts, use prototype embeddings instead. 
    #   Only for testing; setting this on may lead to performance drop.
    exclude_abs: bool
    # Model name for node abstraction, "[PLATFORM]:[MODEL_NAME_OR_PATH]"
    #   e.g., "ollama:llama3.3:latest", "transformers:Voicelab/vlt5-base-keywords", 
    #   "api:google/gemini-2.5-flash"
    abs_name: str | None
    # Cache directory of the abstraction agent, default to os.environ["HF_HOME"].
    abs_cache_dir: Path
    # Model kwargs for the abstraction agent, similar to embed_model_kwargs.
    abs_model_kwargs: Dict
    # Type of abstract. "summary" for summative text and "keyword" for keywords.
    abstract_type: str | None
    # Maximum word count for node abstraction (for LLM prompting so not always work 
    #   depending on how much LLM listens to you). Maximum keyword count for keyword abstract. 
    max_abs_length: int
    # Abstract nodes not higher than this layer will also be retrieved, similar to the 
    #   "collapsed tree" setting in [RAPTOR](https://github.com/parthsarthi03/raptor); 
    #   maybe useful in summative tasks. Default to 0 (to only retrieve leaf nodes). 
    #   Note: generated abstracts occupy tree_top_k slots, so retrieval evaluation results 
    #   would be inaccurate.
    abstract_layer_as_context: int
    # Recreate embeddings and trees even if saved files exist in save_dir.
    #   Warning: your previous saved file will be covered! 
    force_index_from_scratch: bool

    # ==================================== R&A Agent config ===================================
    # Model name for agentic retrieval and QA, "[PLATFORM]:[MODEL_NAME_OR_PATH]"
    #   e.g., "transformers:meta-llama/Llama-3.3-70B-Instruct", "ollama:llama3.3:latest", 
    #   "api:openai/gpt-5-mini"
    qa_name: str
    # Cache directory of the R&A agent, default to os.environ["HF_HOME"].
    qa_cache_dir: Path
    # Model kwargs for the R&A agent, similar to embed_model_kwargs.
    qa_model_kwargs: Dict
    # The expected answer type. This shapes the system instruction to match the task.
    #   "short" for token-level tasks (single-hop and multi-hop qa).
    #   "medium" for passage-level tasks (narrative qa).
    #   "long" for document-level tasks (summarization).
    #   set to "auto" to exclude length-related instructions; specify the length for reproduction.
    answer_type: str
    # Skip retrieval entirely and answer the question directly with the QA model.
    no_retrieval: bool
    # Maximum time of retrieval attempts. The first retrieval is fixed and does not count.
    #   It can also be "auto" to be predicted dynamically by the query hop discriminator.
    max_retrieval_time: int | str
    # Training data directory read by the hop discriminator. Default to "data/".
    #   Corpus files required: musique, 2wikimultihopqa, hotpotqa, multihoprag, nq
    hop_discriminator_training_data_dir: Path
    # Maximum word count for LLM's thoughts or summary text (for LLM prompting 
    #   so not always work depending on how much LLM listens to you). 
    #   You may want to lower this value if you are calling the OpenAI API, 
    #   but lowering this value may cause erroneous response parsing.
    max_response_length: int
    # Multithread QA for efficient reproduction on large corpus. If Ollama models are used, 
    #   it is recommended to set this equal to the environment variable "OLLAMA_NUM_PARALLEL".
    multithreading_qa_batch_size: int
    # Reanswering questions even if the answer result file exists in save_dir.
    #   Warning: your previous saved file will be covered! 
    force_qa_from_scratch: bool

    # ================================ General retrieval config ===============================
    # Document selection mode during retrieval, ("top_k", "threshold"). Unused.
    selection_mode: str
    # Number of retrieved documents if selection_mode="top_k". Note that this is only 
    #   for tree retrieval; the final top-k always depend on rerank_top_k if set.
    #   This is the final top_k only if rerank_top_k is None or rerank_top_k == tree_top_k.
    tree_top_k: int
    # Threshold for retrieving documents if selection_mode="threshold". Unused.
    threshold: float
    # From what layer should tree search start. Set to None to automatically identify. 
    #   Set to 0 to search on the entire leaf set as a flat index.
    start_layer: int | None
    # Metric to calculate vector distance for retrieval, ("cosine", "L1", "L2", "Linf"). 
    distance: str

    # ================================= Hybrid retrieval config ================================
    # If sparse retrieval is enabled (only BM25S is supported for now).
    hybrid_search: bool
    # Rebuild the sparse token vocab even if saved files exist in save_dir/<sparse_method>_<dataset>.
    #   Warning: your previous saved files will be covered! 
    force_sparse_index_from_scratch: bool
    # Number of retrieved documents by sparse keyword search. 
    sparse_top_k: int
    # Threshold for BM25. Unused.
    sparse_threshold: float
    # k in reciprocal rank fusion, only if rerank=False. Default to 60.
    rrf_k: int
    
    # ==================================== Reranking config ===================================
    # If reranking is enabled.
    rerank: bool
    # Model name for reranking, "[PLATFORM]:[MODEL_NAME_OR_PATH]"
    #   e.g., "transformers:Qwen/Qwen3-Reranker-8B", "transformers:BAAI/bge-reranker-large"
    rerank_name: str | None
    # Cache directory of the reranking model, default to os.environ["HF_HOME"].
    rerank_cache_dir: Path | None
    # Model kwargs for the reranker, similar to embed_model_kwargs.
    rerank_model_kwargs: Dict
    # Number of final returned documents by the reranker if selection_mode="top_k".
    rerank_top_k: int | None
    # Threshold for reranking score if selection_mode="threshold". Unused.
    rerank_threshold: float | None
    # Batch size for the reranker. Should be smaller than rerank_top_k. 
    #   Used only when Qwen3-Reranker-8B OOMs as this reranker does not support auto device mapping.
    rerank_batch_size: int

    # ================================ EnterpriseRAG-Bench config ==============================
    # Directory of the benchmark (contains data/documents/test.parquet, data/questions/test.parquet).
    enterprise_data_dir: Path
    # Number of documents to index: every document referenced by a question is kept, the rest of
    #   the budget is sampled uniformly per source type. None = the full corpus (512k documents).
    enterprise_subset_size: int | None
    # Random seed of the distractor sampling (part of the index file name).
    enterprise_subset_seed: int
    # Where the sampled subset is cached (None -> <enterprise_data_dir>/subsets).
    enterprise_subset_cache_dir: Path | None
    # Curated project / codename vocabulary used by the rule-based metadata parsers (None = empty).
    enterprise_project_vocab: Path | None
    # Prefix every chunk with the document title (channel / account / ticket title) after chunking.
    enterprise_chunk_title_prefix: bool
    # Heuristic authority rank per source type stored in abstract nodes' aggregated_metadata.
    enterprise_source_authority: Dict[str, int]
    # Prepend "[doc: id | source | title | date | author]" to every retrieved chunk in the QA context.
    context_metadata_header: bool
    # LLM judge for EnterpriseRAG-Bench answers ("api:gpt-4o-mini" style name, None = disabled).
    judge_name: str | None
    judge_cache_dir: Path | None
    judge_model_kwargs: Dict
    # Parallel judge calls.
    judge_workers: int

    # ============================ Metadata-aware retrieval config ============================
    # Retrieval pipeline. "legacy" = top-down tree traversal + BM25 fused by RRF / reranker
    #   (bit-for-bit the original behaviour). "hybrid_score" = candidate generation over the tree
    #   followed by the metadata-aware hybrid score
    #   alpha*dense + beta*bm25 + gamma*metadata + delta*level - lambda*redundancy (MMR selection).
    retrieve_mode: str
    # Candidate generation for "hybrid_score": "collapsed" = dense top-N over every node of every
    #   layer (cached normalised matrix) U BM25 top-N leaves; "traversal" = nodes visited by the
    #   legacy top-down traversal U BM25 hits.
    candidate_mode: str
    candidate_dense_top_n: int
    candidate_sparse_top_n: int
    # Weights of the hybrid score: {"alpha","beta","gamma","delta","lambda"}.
    score_weights: Dict[str, float]
    # Per-query normalisation of the dense / BM25 terms over the candidate pool, ("minmax", "rank").
    score_norm: str
    # dtype of the cached dense matrix ("float32" or "float16").
    dense_index_dtype: str
    # Field weights of S_metadata (only fields present in the parsed query count).
    metadata_field_weights: Dict[str, float]
    # Days added on both sides of a query time window before matching document dates.
    time_tolerance_days: int
    # S_level: question_type -> {"0": w, "1": w, "2+": w} preference for leaf / document / cluster nodes.
    level_preference: Dict[str, Dict[str, float]]
    # Hard filter (arm D): drop nodes that contradict the high-precision constraints of the query.
    metadata_filter: bool
    hard_filter_fields: Sequence[str]
    hard_time_tolerance_days: int
    # Query understanding: "none" (no constraints), "rules" (dates / vocabulary / ticket regex),
    #   "llm" (one JSON call to query_name, rules as fallback + augmentation, cached on disk).
    query_understanding: str
    # Chat model for query understanding (None -> reuse qa_name); cache dir (None -> <save_dir>/query_cache).
    query_name: str | None
    query_cache_dir: Path | None
    query_model_kwargs: Dict
    # Abstract nodes contribute their source documents to retrieved_doc_ids only when they cover
    #   at most this many documents (None = no cap).
    source_max_abstract_docs: int | None
    # Store per-query parses and candidate sub-scores in the results file.
    save_retrieval_diagnostics: bool
    # Metadata-in-text index (arm B): write a metadata header into every chunk / abstract before
    #   embedding and BM25 indexing. Index files get a "_metatext" tag.
    enterprise_chunk_metadata_prefix: bool
    # Evaluate only the "dev" or "test" question split (None = all questions).
    enterprise_split: str | None
    enterprise_split_file: Path
    # Suffix for result / log / judge-cache file names so several variants of one config coexist.
    run_tag: str | None

    # ===================================== Other config ======================================
    # Set of evaluation metrics, ("em", "f1", "rouge", "recall", "answerrate",
    #   "docrecall", "extradocs", "llmjudge"). The last three need EnterpriseRAG-Bench labels.
    #   Default to "all" to use all supported metrics.
    evaluation_metrics: str | Sequence[str]
    # Save directory of everything intermediate: tree, BM25 vocab, QA results, etc. 
    #   Set to None to skip saving for a single run.
    save_dir: Path
    # Path of your log files. 
    log_path: Path
    # Whether to output the detailed information (e.g., LLM full responses) during QA.
    verbose: bool


# Default configuration. Each conf/<setting>.py applies custom setting on top of this.   
conf = Config(
    dataset="musique",
    data_dir="./data",
    read_local_pdf=None,
    pdf_batch_size=1,
    test_samples=-1,
    force_split=False,
    tokenizer="cl100k_base",
    max_tokens_per_chunk=100,
    chunking="semantic",
    semantic_threshold=90,
    semantic_threshold_type="percentile",
    semantic_distance="cosine",
    short_chunk_tokens=30,
    semantic_merge_threshold=None,
    semantic_embed_batch_size=64,
    
    embed_name="api:text-embedding-3-small",
    embed_cache_dir=None,
    embed_model_kwargs={},
    
    tree_builder="exact",
    passage_as_tree=False,
    reorganize_leaf=False,
    max_num_children=40,
    tree_build_diagnostics=False,
    bucket_size=None,
    bucket_max_fanout=32,
    bucket_sample_size=8192,
    bucket_kmeans_iters=5,
    tree_save_chunk_size=200000,
    partition_ratio=1.,
    hnsw_top_k=32,
    hnsw_m=16,
    hnsw_ef_construction=200,
    hnsw_ef_search=64,

    exclude_abs=False,
    abs_name="ollama:llama3.3:latest",
    abs_cache_dir=None,
    abs_model_kwargs={},
    abstract_type="summary",
    max_abs_length=100,
    abstract_layer_as_context=0,
    force_index_from_scratch=False,
    
    qa_name="ollama:llama3.3:latest",
    qa_cache_dir=None,
    qa_model_kwargs={},
    answer_type="short",
    no_retrieval=False,
    max_retrieval_time=3,
    hop_discriminator_training_data_dir="./data",
    max_response_length=500,
    multithreading_qa_batch_size=int(os.environ["OLLAMA_NUM_PARALLEL"]) if "OLLAMA_NUM_PARALLEL" in os.environ else -1,
    force_qa_from_scratch=False,

    selection_mode="top_k",
    tree_top_k=5,
    threshold=0.5,
    start_layer=None,
    distance="cosine",

    hybrid_search=False,
    force_sparse_index_from_scratch=False,
    sparse_top_k=5,
    sparse_threshold=10,
    rrf_k=60,

    rerank=False,
    rerank_name=None,
    rerank_cache_dir=None,
    rerank_model_kwargs={},
    rerank_top_k=5,
    rerank_threshold=None,
    rerank_batch_size=-1,

    enterprise_data_dir="./data/enterpriseRAG-Bench",
    enterprise_subset_size=5000,
    enterprise_subset_seed=42,
    enterprise_subset_cache_dir=None,
    enterprise_project_vocab="./conf/enterprise_projects.json",
    enterprise_chunk_title_prefix=True,
    enterprise_source_authority={
        "confluence": 5, "google_drive": 4, "jira": 3, "linear": 3, "github": 3,
        "gmail": 2, "fireflies": 2, "hubspot": 2, "slack": 1,
    },
    context_metadata_header=False,
    judge_name=None,
    judge_cache_dir=None,
    judge_model_kwargs={},
    judge_workers=8,

    retrieve_mode="legacy",
    candidate_mode="collapsed",
    candidate_dense_top_n=100,
    candidate_sparse_top_n=100,
    score_weights={"alpha": 1.0, "beta": 0.5, "gamma": 0.5, "delta": 0.3, "lambda": 0.3},
    score_norm="minmax",
    dense_index_dtype="float32",
    metadata_field_weights={
        "ticket_keys": 3.0, "time": 2.0, "source_type": 2.0, "projects": 2.0,
        "entities": 1.5, "people": 1.5, "channels": 1.0,
    },
    time_tolerance_days=7,
    level_preference={
        "high_level": {"0": 0.2, "1": 0.6, "2+": 1.0},
        "completeness": {"0": 0.2, "1": 0.6, "2+": 1.0},
        "project_related": {"0": 0.8, "1": 0.8, "2+": 0.4},
        "miscellaneous": {"0": 0.8, "1": 0.8, "2+": 0.4},
        "default": {"0": 1.0, "1": 0.5, "2+": 0.0},
    },
    metadata_filter=False,
    hard_filter_fields=["ticket_keys", "time", "source_type"],
    hard_time_tolerance_days=14,
    query_understanding="none",
    query_name=None,
    query_cache_dir=None,
    query_model_kwargs={},
    source_max_abstract_docs=3,
    save_retrieval_diagnostics=False,
    enterprise_chunk_metadata_prefix=False,
    enterprise_split=None,
    enterprise_split_file="./conf/enterprise_splits_s42.json",
    run_tag=None,

    evaluation_metrics="all",

    save_dir="./output",
    log_path="./log",
    verbose=False,
)

def read_config(conf_name: str = None) -> Config:
    loaded_conf = conf.copy()
    if conf_name is None:
        loaded_conf["config"] = "default"
        return loaded_conf

    config_path = "./conf"
    if os.path.exists(os.path.join(config_path, f"{conf_name}.py")):
        module = importlib.import_module(f"conf.{conf_name}")
        
        if hasattr(module, 'conf'):
            loaded_conf.update(module.conf)
        else:
            raise AttributeError(f"There is no importable config in the config file \"{conf_name}\". ")
        
        loaded_conf["config"] = conf_name
    else:
        raise FileNotFoundError(f"Config file \"{conf_name}\" not found. ")

    return loaded_conf


def _parse_override_value(value: str):
    lowered_value = value.lower()
    if lowered_value == "true":
        return True
    if lowered_value == "false":
        return False
    if lowered_value == "none":
        return None

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def apply_config_overrides(conf: Config, extra_args: Sequence[str]) -> Config:
    i = 0
    while i < len(extra_args):
        arg = extra_args[i]
        if not arg.startswith("--"):
            raise ValueError(f"Unsupported argument format: {arg}")

        arg_body = arg[2:]
        if "=" in arg_body:
            key, raw_value = arg_body.split("=", maxsplit=1)
        else:
            key = arg_body
            if key not in conf:
                raise ValueError(f'Unknown config override "--{key}".')

            if isinstance(conf[key], bool):
                if i + 1 < len(extra_args) and not extra_args[i + 1].startswith("--"):
                    i += 1
                    raw_value = extra_args[i]
                else:
                    raw_value = "true"
            else:
                if i + 1 >= len(extra_args) or extra_args[i + 1].startswith("--"):
                    raise ValueError(f'Missing value for "--{key}".')
                i += 1
                raw_value = extra_args[i]

        if key not in conf:
            raise ValueError(f'Unknown config override "--{key}".')

        conf[key] = _parse_override_value(raw_value)
        i += 1

    return conf
