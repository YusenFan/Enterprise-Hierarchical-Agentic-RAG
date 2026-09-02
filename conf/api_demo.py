# OpenAI API demo: embedding, abstraction and QA all go through the OpenAI-compatible
# endpoint configured in ".env" (OPENAI_API_KEY, and OPENAI_BASE_URL for non-official endpoints).
# Official OpenAI model ids carry no "openai/" prefix.
conf = {
    "dataset": "musique",
    "test_samples": 10,

    "embed_name": "api:text-embedding-3-small",
    "abs_name": "api:gpt-4o-mini",
    "abstract_type": "summary",
    "qa_name": "api:gpt-4o-mini",

    "max_retrieval_time": 3,
    "tree_top_k": 10,
    "hybrid_search": True,
    "sparse_top_k": 10,
    "rerank_top_k": 5,

    "max_response_length": 200,
    "force_index_from_scratch": True,
    "verbose": True,
}
