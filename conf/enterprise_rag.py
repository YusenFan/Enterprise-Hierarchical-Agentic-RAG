# EnterpriseRAG-Bench with metadata-aware ingestion. Embedding, abstraction, QA and the judge all
# go through the OpenAI-compatible endpoint configured in ".env" (OPENAI_API_KEY / OPENAI_BASE_URL).
# Index ~5k documents (all question-referenced documents + uniformly sampled distractors).
conf = {
    "dataset": "enterprise_rag",
    "enterprise_subset_size": 5000,
    "enterprise_subset_seed": 42,
    "force_split": True,                    # corpus documents are chunked at index time
    "chunking": "semantic",
    "max_tokens_per_chunk": 100,

    "embed_name": "api:text-embedding-3-small",
    "abs_name": "api:gpt-4o-mini",
    "abstract_type": "summary",
    "max_abs_length": 100,
    "qa_name": "api:gpt-4o-mini",
    "judge_name": "api:gpt-4o-mini",

    "tree_builder": "exact",
    "reorganize_leaf": False,               # layer 1 = one abstract per document, higher layers cross-document
    "max_num_children": 40,

    "answer_type": "medium",
    "max_retrieval_time": 1,
    "max_response_length": 300,
    "tree_top_k": 10,
    "hybrid_search": True,
    "sparse_top_k": 10,
    "rerank_top_k": 10,
    "context_metadata_header": True,

    "evaluation_metrics": ["docrecall", "extradocs", "llmjudge", "f1", "rouge", "answerrate"],
    "multithreading_qa_batch_size": 8,
    "verbose": False,
}
