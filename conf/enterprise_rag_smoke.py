# Offline smoke test of the EnterpriseRAG-Bench pipeline: hashed embeddings, fake abstracts and a
# fake QA model, ~60 documents (all gold documents of the first questions are still kept).
conf = {
    "dataset": "enterprise_rag",
    "enterprise_subset_size": 800,          # must be >= the number of gold documents (722)
    "enterprise_subset_seed": 42,
    "test_samples": 10,
    "force_split": True,
    "chunking": "static",
    "max_tokens_per_chunk": 100,

    "embed_name": "hash:bow",
    "abs_name": "fake:abstract",
    "qa_name": "fake:qa",
    "judge_name": None,

    "tree_builder": "exact",
    "reorganize_leaf": False,
    "max_num_children": 40,

    "answer_type": "medium",
    "max_retrieval_time": 1,
    "tree_top_k": 10,
    "hybrid_search": True,
    "sparse_top_k": 10,
    "rerank_top_k": 10,
    "context_metadata_header": True,
    "query_understanding": "rules",         # offline query parsing (no LLM)

    "evaluation_metrics": ["docrecall", "docmrr", "docndcg", "extradocs", "f1", "answerrate"],
    "multithreading_qa_batch_size": -1,
    "save_dir": "./output/smoke",
    "force_index_from_scratch": True,
    "verbose": False,
}
