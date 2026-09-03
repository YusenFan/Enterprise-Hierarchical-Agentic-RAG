import os
import json
import logging

from itertools import chain
from typing import Any, Dict, List, Set
from tqdm import tqdm

import numpy as np

from .enterprise_rag import load_enterprise_rag
from .metadata import Document, ProjectVocabulary, document_text_header, get_parser
from .pdf import load_local_pdf_data
from .utils import split_text, split_sentences, chunk_sentences_semantic

dataset_pool = (
    "nq",
    "popqa",
    "narrativeqa",
    "hotpotqa",
    "2wikimultihopqa",
    "musique",
    "multihoprag",
    "infinitybench_longbook",
    "qmsum",
    "wcep",
    "enterprise_rag",
)


class DataManager:

    def __init__(self,
                 dataset_name: str, 
                 data_dir: str = "./data",
                 test_samples: int = -1,
                 local_pdf: str | None = None,
                 enterprise_kwargs: Dict[str, Any] | None = None,
                 **pdf_kwargs) -> None:
        
        self.dataset_name: str = dataset_name.lower()
        self.local_pdf = local_pdf
        if not self.local_pdf and self.dataset_name not in dataset_pool:
            raise NotImplementedError(f"Dataset {self.dataset_name} is currently not supported.")

        # EnterpriseRAG-Bench: metadata-bearing documents (see src/metadata).
        self.is_enterprise: bool = self.dataset_name == "enterprise_rag"
        self.enterprise_kwargs: Dict[str, Any] = dict(enterprise_kwargs or {})
        self.documents: List[Document] | None = None          # one Document per corpus document
        self.document_registry: Dict[str, Document] | None = None
        self.chunk_local_metadata: List[List[Dict]] | None = None
        self.gold_doc_ids: List[List[str]] | None = None       # expected_doc_ids per question
        self.answer_facts: List[List[str]] | None = None
        self.question_types: List[str] | None = None
        self.question_ids: List[str] | None = None
        self.subset_stats: Dict[str, Any] = {}
        self.project_vocab: ProjectVocabulary | None = None

        self.data: List[Dict] | None = None
        self.data_path: str | None = None 
        self.corpus: str | List[str] | List[Dict[str, Any]] | None = None
        self.load_data(data_dir, **pdf_kwargs)
        self._apply_enterprise_split()

        # ------------------- FOR QA TESTING ONLY -------------------
        if test_samples > 0:
            if self.data is not None:
                self.data = self.data[:test_samples]
            # The enterprise corpus size is controlled by the subset, not by test_samples:
            # every gold document must stay in the index for the questions that are kept.
            if self.corpus is not None and not self.is_enterprise:
                self.corpus = self.corpus[:test_samples]
        # ------------------- FOR QA TESTING ONLY -------------------

        self.gold_docs: List[List[str]] | None = []
        self.gold_nodes_id: List[List[int]] | None = []
        self.get_gold_docs()

        self.gold_answers: List[Set[str]] | None = []
        self.get_gold_answers()
        
        self.all_queries: List[str] = []
        self.query_to_doc_ids: List[int] = []
        self.all_text_ids: List[str] = []
        self.all_passages: List[str] | List[List[str]] = []
        self.preprocess()

    def _apply_enterprise_split(self) -> None:
        """Keep only the questions of the configured dev / test split (enterprise_kwargs["split"])."""
        split = (self.enterprise_kwargs or {}).get("split")
        if not self.is_enterprise or not split or self.data is None:
            return
        split_file = self.enterprise_kwargs.get("split_file")
        if not split_file or not os.path.exists(split_file):
            raise FileNotFoundError(
                f'Question split file "{split_file}" not found. Run "python experiments/make_splits.py" first.'
            )
        with open(split_file, "r", encoding="utf-8") as f:
            splits = json.load(f)
        if split not in splits:
            raise ValueError(f'Split "{split}" not in {sorted(k for k in splits if k != "meta")} of "{split_file}".')
        keep = set(splits[split])
        before = len(self.data)
        self.data = [sample for sample in self.data if sample.get("question_id") in keep]
        self.subset_stats = dict(self.subset_stats or {}, split=split, split_questions=len(self.data),
                                 all_questions=before)
        logging.info(f'Enterprise split "{split}": {len(self.data)} / {before} questions kept.')

    def load_data(self, data_dir: str = "./data", **pdf_kwargs) -> None:
        try:
            if self.local_pdf:
                self.data_path = str(data_dir)
                self.data = load_local_pdf_data(data_dir, self.local_pdf, **pdf_kwargs)
                self.corpus = None
            elif self.dataset_name in  ("hotpotqa", "2wikimultihopqa", "musique", 
                                        "nq", "popqa", "multihoprag", "narrativeqa"):
                self.data_path = os.path.join(data_dir, self.dataset_name) + ".json"
                corpus_path = os.path.join(data_dir, f"{self.dataset_name}_corpus") + ".json"
                self.corpus = json.load(open(corpus_path, "r"))
                self.data = json.load(open(self.data_path, "r"))
            elif self.dataset_name in ("infinitybench_longbook",):
                self.data_path = os.path.join(data_dir, self.dataset_name) + ".jsonl"
                self.data = []
                with open(self.data_path, 'rb') as file:
                    for line in file:
                        self.data.append(json.loads(line))
            elif self.dataset_name in ("qmsum",):
                self.data_path = os.path.join(data_dir, self.dataset_name) + ".jsonl"
                self.data = []
                with open(self.data_path, 'r') as file:
                    for line in file:
                        self.data.append(json.loads(line))
            elif self.dataset_name in ("wcep",):
                self.data_path = os.path.join(data_dir, self.dataset_name) + ".txt"
                self.data = []
                with open(self.data_path, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        self.data.append(json.loads(line))
            elif self.is_enterprise:
                kw = self.enterprise_kwargs
                self.data_path = kw.get("data_dir") or os.path.join(data_dir, "enterpriseRAG-Bench")
                self.corpus, self.data, self.subset_stats = load_enterprise_rag(
                    self.data_path,
                    subset_size=kw.get("subset_size", 5000),
                    seed=kw.get("seed", 42),
                    cache_dir=kw.get("cache_dir"),
                    force=bool(kw.get("force_rebuild", False)),
                )
                self.project_vocab = ProjectVocabulary.load(kw.get("project_vocab"))
            else:
                raise NotImplementedError
        except FileNotFoundError:
            if self.local_pdf:
                raise
            raise FileNotFoundError(f"{self.data_path} not found.")

    def get_gold_docs(self) -> None:
        """
        Get supporting documents for retrieval evaluation. Depending on the dataset format, 
        gold documents can be either a list of strings or a list of list of strings.
        For EnterpriseRAG-Bench the labels are document ids (`gold_doc_ids`), not texts.
        """
        if self.is_enterprise:
            self.gold_doc_ids = [list(sample.get("expected_doc_ids", [])) for sample in self.data]
            self.answer_facts = [list(sample.get("answer_facts", [])) for sample in self.data]
            self.question_types = [sample.get("question_type") for sample in self.data]
            self.question_ids = [sample.get("question_id") for sample in self.data]
            self.gold_docs = None
            self.gold_nodes_id = None
            return
        for sample in self.data:
            gold_node_id = []
            if 'supporting_facts' in sample:  # hotpotqa, 2wikimultihopqa
                gold_title = set([item[0] for item in sample['supporting_facts']])
                gold_title_and_content_list = []
                for i, item in enumerate(sample['context']):
                    if item[0] in gold_title:
                        gold_title_and_content_list.append(item)
                        gold_node_id.append(i)
                if self.dataset_name.startswith('hotpotqa'):
                    gold_doc = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
                else:
                    gold_doc = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
            elif 'contexts' in sample:
                gold_doc = []
                for i, item in enumerate(sample['contexts']):
                    if item['is_supporting']:
                        gold_doc.append(item['title'] + '\n' + item['text'])
                        gold_node_id.append(i)
            elif 'paragraphs' in sample:
                gold_paragraphs = []
                for i, item in enumerate(sample['paragraphs']):
                    if 'is_supporting' in item and item['is_supporting'] is False:
                        continue
                    gold_paragraphs.append(item)
                    if 'idx' in item:
                        gold_node_id.append(item['idx'])
                    else:
                        gold_node_id.append(i)
                gold_doc = [item['title'] + '\n' + (item['text'] if 'text' in item else item['paragraph_text']) for item in gold_paragraphs]
            elif 'evidence_list' in sample: # multihoprag
                gold_doc = [evidence['fact'] for evidence in sample['evidence_list']]
                gold_node_id = -1
            else:
                print(f'Warning: dataset "{self.dataset_name}" seems to be lack of labels for retrieval evaluation.') 
                self.gold_docs = None
                self.gold_nodes_id = None
                return

            gold_doc = list(set(gold_doc))
            self.gold_docs.append(gold_doc)
            self.gold_nodes_id.append(gold_node_id)

    def get_gold_answers(self) -> None:
        """
        Get ground-truth answers for QA evaluation.
        If your custom dataset has 'answer' 'gold_ans' 'golden_answers' or 'reference' field, 
        they will be automatically read as gold answers.
        Otherwise, you may need to add a conditional branch to fit your dataset format.
        """
        for sample in self.data:
            gold_ans = None

            if 'answer' in sample:
                gold_ans = sample['answer']
            elif 'gold_ans' in sample:
                gold_ans = sample['gold_ans']
            elif 'reference' in sample:
                gold_ans = sample['reference']
            elif 'obj' in sample:
                gold_ans = set(
                    [sample['obj']] + [sample['possible_answers']] + [sample['o_wiki_title']] + [sample['o_aliases']])
                gold_ans = list(gold_ans)
            elif 'golden_answers' in sample:
                gold_ans = sample['golden_answers']
            elif 'gold_answer' in sample:  # enterprise_rag
                gold_ans = sample['gold_answer']
            elif self.dataset_name in ("qmsum"):
                gold_ans = sample['general_query_list'][0]['answer']
            elif self.dataset_name in ("wcep"):
                word_num = len(" ".join(sample["document"]).split())
                if word_num <= 6000:
                    continue
                gold_ans = sample["summary"]
            else:
                print(f'Warning: dataset "{self.dataset_name}" seems to be lack of labels for QA evaluation.') 
                self.gold_answers = None
                return

            if isinstance(gold_ans, str):
                gold_ans = [gold_ans]
            assert isinstance(gold_ans, list)
            gold_ans = set(gold_ans)
            if 'answer_aliases' in sample:
                gold_ans.update(sample['answer_aliases'])

            self.gold_answers.append(gold_ans)

    def _preprocess_enterprise(self) -> None:
        """
        Parse every corpus row with its source-type parser: the Document registry becomes the
        authoritative metadata source, `all_text_ids` holds the document ids (aligned with
        `all_passages`, so `document_index` in the tree still indexes both lists).
        """
        self.documents = []
        self.document_registry = {}
        if self.enterprise_kwargs.get("vocab_from_hubspot", True):
            # HubSpot titles are clean account names: let every source match them as entities.
            accounts = {row["title"].strip(): [row["title"].strip()] for row in self.corpus
                        if row.get("source_type") == "hubspot" and row.get("title")
                        and 3 <= len(row["title"].strip()) <= 60}
            vocab = self.project_vocab or ProjectVocabulary()
            self.project_vocab = ProjectVocabulary(vocab.projects, {**accounts, **vocab.entities})
        bar = tqdm(self.corpus, desc="parsing metadata")
        for row in bar:
            doc_id = row["doc_id"]
            if doc_id in self.document_registry:
                continue
            parser = get_parser(row["source_type"])
            text = parser.normalize_content(row.get("content") or "")
            doc = parser.parse(doc_id, row["source_type"], row.get("title") or "", text, vocab=self.project_vocab)
            self.documents.append(doc)
            self.document_registry[doc_id] = doc
            self.all_text_ids.append(doc_id)
            self.all_passages.append(text)  # List[str]; chunked later by split_dataset()
        bar.close()

    def finalize_chunks(self, title_prefix: bool = True, metadata_prefix: bool = False) -> None:
        """
        Called once the passages are chunked (List[List[str]]): computes chunk-level
        `local_metadata` and optionally prefixes every chunk with the document title so that
        channel / account / ticket titles reach every chunk (tree text and BM25 text stay identical).
        `metadata_prefix` (metadata-in-text index) writes a "[source | title | date | who | ...]"
        header instead of the bare title.
        """
        if not self.documents:
            return
        if isinstance(self.all_passages, str):
            self.all_passages = [[self.all_passages]]
        elif self.all_passages and isinstance(self.all_passages[0], str):
            self.all_passages = [[passage] for passage in self.all_passages]
        self.chunk_local_metadata = []
        for doc, chunks in zip(self.documents, self.all_passages):
            parser = get_parser(doc.source_type)
            state: Dict[str, Any] = {}
            self.chunk_local_metadata.append([parser.local_metadata(chunk, doc, state) for chunk in chunks])
            if metadata_prefix:
                header = document_text_header(doc)
                chunks[:] = [f"{header}\n{chunk}" for chunk in chunks]
            elif title_prefix and doc.title:
                chunks[:] = [f"{doc.title}\n{chunk}" for chunk in chunks]
            doc.num_chunks = len(chunks)

    def preprocess(self):
        if self.is_enterprise:
            self._preprocess_enterprise()
            bar = tqdm(self.data, desc="preprocessing query")
            for i, sample_text in enumerate(bar):
                self.all_queries.append(sample_text['question'])
                self.query_to_doc_ids.append(i)
            bar.close()
            return

        if self.corpus is not None: # musique, hotpotqa, 2wikimultihopqa, nq, popqa, narrativeqa, multihoprag
            bar = tqdm(self.corpus, desc="preprocessing text") 
        else: # qmsum, wcep, infinitybench_longbook
            bar = tqdm(self.data, desc="preprocessing text") 

        if self.dataset_name in ("narrativeqa", "infinitybench_longbook"):
            doc_dict: Dict[str, str|List[str]] = {}

        # Read documents (passages) and ids. 
        # Depending on the dataset format, passages can be either a list of strings 
        # or a list of list of strings. This affects the subsequent `split_text()` process. 
        # self.all_text_ids: List[str] contains all document ids.
        # self.all_passages: List[str] or List[List[str]] contains all documents. 
        doc_idx = 0
        for sample_text in bar:
            if self.dataset_name in ('musique', 'hotpotqa', '2wikimultihopqa', 'nq', 'popqa'):
                self.all_text_ids.append(str(doc_idx))
                self.all_passages.append(sample_text['title'] + "\n" + sample_text['text'])
            if self.dataset_name in ("multihoprag",):
                self.all_text_ids.append(str(doc_idx))
                self.all_passages.append(sample_text['title'] + "\n" + sample_text['body'])
            elif self.dataset_name in ('qmsum',):
                def clean_qmsum_data(text: str) -> str:
                    text = text.replace('{vocalsound}', '')
                    text = text.replace('{disfmarker}', '')
                    text = text.replace('A_M_I_', 'ami')
                    text = text.replace('L_C_D_', 'lcd')
                    text = text.replace('P_M_S', 'pms')
                    text = text.replace('T_V_', 'tv')
                    text = text.replace('{pause}', '')
                    text = text.replace('{nonvocalsound}', '')
                    text = text.replace('{gap}', '')
                    text = text.replace('  ', ' ')
                    return text
                self.all_text_ids.append(str(doc_idx))
                passage = "\n".join(["Speaker: " + i["speaker"] + "\n" + "Content: " + i["content"]
                                    for i in sample_text['meeting_transcripts']])
                self.all_passages.append(clean_qmsum_data(passage))
            elif self.dataset_name in ('wcep',):
                word_num = len(" ".join(sample_text["document"]).split())
                if word_num <= 6000:
                    # self.all_queries = self.all_queries[:-1]
                    continue
                self.all_passages.append(' '.join(sample_text["document"]))
                self.all_queries.append("Summarize the contents of this news event.")
                self.query_to_doc_ids.append(doc_idx)
            elif self.dataset_name in ("narrativeqa",):
                doc_id = sample_text['idx'].split('_', maxsplit=1)[0]
                if doc_id in doc_dict:
                    doc_dict[doc_id].append(sample_text['title'] + "\n" + sample_text['text'])
                else:
                    doc_dict[doc_id] = [sample_text['title'] + "\n" + sample_text['text']]
            elif self.dataset_name in ("infinitybench_longbook",):
                doc_id = sample_text['context'][:100]
                doc_dict[doc_id] = sample_text['context']
            elif self.local_pdf:
                self.all_text_ids.append(sample_text["title"])
                self.all_passages.append([
                    sample_text["title"] + "\n" + chunk
                    for chunk in sample_text["chunks"]
                ])
                
            doc_idx += 1

        if self.dataset_name in ("narrativeqa", "infinitybench_longbook"):
            self.all_text_ids = list(doc_dict.keys())
            self.all_passages = list(doc_dict.values())
        bar.close()
        
        # Read user queries and map them to document ids. 
        # If your custom dataset has 'query' or 'question' field, they will be automatically read.
        # Otherwise, you may need to add a conditional branch to fit your dataset format.
        # self.all_queries: List[str] contains all user queries. 
        # self.query_to_doc_ids: List[int] contains the mapping from each query to the document index.
        bar = tqdm(self.data, desc="preprocessing query")
        for i, sample_text in enumerate(bar):
            if self.local_pdf:
                break
            elif self.dataset_name in ('narrativeqa',):
                self.all_queries.append(sample_text['question'])
                idx = self.all_text_ids.index(sample_text['document']['id'])
                self.query_to_doc_ids.append(idx)
            elif self.dataset_name in ('infinitybench_longbook',):
                self.all_queries.append(sample_text['input'])
                idx = self.all_text_ids.index(sample_text['context'][:100])
                self.query_to_doc_ids.append(idx)
            elif self.dataset_name in ('qmsum',):
                self.all_queries.append(sample_text['general_query_list'][0]['query'])
                self.query_to_doc_ids.append(i)
            elif 'query' in sample_text:
                self.all_queries.append(sample_text['query'])
                self.query_to_doc_ids.append(i)
            elif 'question' in sample_text:
                self.all_queries.append(sample_text['question'])
                self.query_to_doc_ids.append(i)
        bar.close()
        
    def split_text(
        self,
        chunking: str = "static",
        tokenizer: str | None = None,
        max_tokens: int = 100,
        embed_model=None,
        semantic_threshold: float = 90,
        semantic_threshold_type: str = "percentile",
        distance_metric: str = "cosine",
        short_chunk_tokens: int = 0,
        merge_threshold: float | None = None,
        embed_batch_size: int = 64,
        distance_recorder: Dict | None = None,
    ) -> None:
        """
        Split text into chunks according to dataset format and chunking method.

        "static": pack sentences greedily up to max_tokens (see `utils.split_text`).
        "semantic": embed every sentence with `embed_model` and cut where consecutive
            sentences drift apart (see `utils.chunk_sentences_semantic`). All sentences of
            the corpus are embedded in one batched pass before chunking each document.
        """
        if chunking == "static":
            if isinstance(self.all_passages, str):
                self.all_passages = split_text(self.all_passages, tokenizer, max_tokens) # List[str]
            elif isinstance(self.all_passages, List) and isinstance(self.all_passages[0], str): 
                self.all_passages = [split_text(t, tokenizer, max_tokens) for t in self.all_passages] # List[List[str]]
            elif isinstance(self.all_passages[0], List):
                self.all_passages = [split_text("\n".join(psg), tokenizer, max_tokens) for psg in self.all_passages] # List[List[str]]
        elif chunking == "semantic":
            if embed_model is None:
                raise ValueError('Semantic chunking requires an embedding model (conf["embed_model"]).')
            self._split_text_semantic(
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                embed_model=embed_model,
                semantic_threshold=semantic_threshold,
                semantic_threshold_type=semantic_threshold_type,
                distance_metric=distance_metric,
                short_chunk_tokens=short_chunk_tokens,
                merge_threshold=merge_threshold,
                embed_batch_size=embed_batch_size,
                distance_recorder=distance_recorder,
            )
        else:
            raise ValueError(f'Unsupported chunking method "{chunking}". Expected "static" or "semantic".')

    def _split_text_semantic(
        self,
        tokenizer,
        max_tokens: int,
        embed_model,
        semantic_threshold: float,
        semantic_threshold_type: str,
        distance_metric: str,
        short_chunk_tokens: int,
        merge_threshold: float | None,
        embed_batch_size: int,
        distance_recorder: Dict | None,
    ) -> None:
        # 1) Normalise the three passage formats to a list of document strings.
        if isinstance(self.all_passages, str):
            documents, single_document = [self.all_passages], True
        elif isinstance(self.all_passages, List) and isinstance(self.all_passages[0], str):
            documents, single_document = list(self.all_passages), False
        elif isinstance(self.all_passages[0], List):
            documents, single_document = ["\n".join(psg) for psg in self.all_passages], False
        else:
            raise ValueError("Unsupported passage format for splitting.")

        # 2) Sentence-split every document.
        document_sentences = [split_sentences(document) for document in documents]
        all_sentences = list(chain.from_iterable(document_sentences))

        # 3) Embed all sentences of the corpus in mini-batches (one pass, not one call per document).
        batch_size = max(int(embed_batch_size), 1)
        embeddings = []
        bar = tqdm(range(0, len(all_sentences), batch_size), desc="embedding sentences")
        for start in bar:
            batch = all_sentences[start: start + batch_size]
            batch_embeddings = np.asarray(embed_model.embed_batch(batch), dtype=np.float32)
            if batch_embeddings.ndim == 1:
                batch_embeddings = batch_embeddings.reshape(len(batch), -1)
            if batch_embeddings.shape[0] != len(batch):
                raise ValueError(
                    f"Embedding model returned {batch_embeddings.shape[0]} embeddings "
                    f"for {len(batch)} sentences."
                )
            embeddings.append(batch_embeddings)
        bar.close()
        all_embeddings = np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 0), dtype=np.float32)

        # 4) Chunk each document on its own sentences and reassemble the original format.
        chunked_documents = []
        offset = 0
        bar = tqdm(document_sentences, desc="semantic chunking")
        for sentences in bar:
            sentence_embeddings = all_embeddings[offset: offset + len(sentences)]
            offset += len(sentences)
            recorder = {} if distance_recorder is not None else None
            chunks = chunk_sentences_semantic(
                sentences,
                sentence_embeddings,
                tokenizer,
                max_tokens,
                semantic_threshold,
                semantic_threshold_type=semantic_threshold_type,
                distance_metric=distance_metric,
                short_chunk_tokens=short_chunk_tokens,
                merge_threshold=merge_threshold,
                distance_recorder=recorder,
            )
            if distance_recorder is not None:
                distance_recorder.setdefault("documents", []).append(recorder)
            chunked_documents.append(chunks)
        bar.close()

        self.all_passages = chunked_documents[0] if single_document else chunked_documents # List[str] | List[List[str]]

    def get_documents(self) -> List[str]:
        """Get a flattened document list (for sparse retrieval)."""
        if isinstance(self.all_passages[0], List):
            docs = []
            for passage in self.all_passages:
                docs.extend(passage)
        else:
            docs = self.all_passages
        return docs
    
    def __len__(self) -> int:
        return len(self.data) if self.data is not None else 0


def enterprise_kwargs_from_conf(conf: Dict) -> Dict[str, Any]:
    """DataManager(enterprise_kwargs=...) from the enterprise_* config keys."""
    return {
        "data_dir": conf.get("enterprise_data_dir"),
        "subset_size": conf.get("enterprise_subset_size", 5000),
        "seed": conf.get("enterprise_subset_seed", 42),
        "cache_dir": conf.get("enterprise_subset_cache_dir"),
        "project_vocab": conf.get("enterprise_project_vocab"),
        "split": conf.get("enterprise_split"),
        "split_file": conf.get("enterprise_split_file"),
    }


def split_dataset(data: DataManager, conf: Dict) -> None:
    """
    Decide whether `data.all_passages` must be split for `conf` and split it in place.
    Shared by index.py / main.py / qa.py / eval.py. Mirrors the three data formats:
      - str: a single long document, always split (passage_as_tree is forced on).
      - List[str]: documents, split if passage_as_tree or force_split.
      - List[List[str]]: preset chunks, re-split only if force_split.
    Semantic chunking needs conf["embed_model"]; when it is missing (e.g. eval.py or
    "no_retrieval" mode, where the chunks are never consumed) static splitting is used.
    """
    if isinstance(data.all_passages, str):
        conf["passage_as_tree"] = True
        conf["force_split"] = True
        need_split = True
    elif isinstance(data.all_passages, List) and isinstance(data.all_passages[0], str):
        need_split = bool(conf["passage_as_tree"] or conf["force_split"])
        if need_split:
            conf["force_split"] = True
    elif isinstance(data.all_passages[0], List):
        need_split = bool(conf["force_split"])
    else:
        need_split = False
    if not need_split:
        return

    chunking = conf.get("chunking", "static")
    embed_model = conf.get("embed_model")
    if chunking == "semantic" and embed_model is None:
        logging.warning(
            "Semantic chunking requires an embedding model but none is loaded; "
            "falling back to static splitting."
        )
        chunking = "static"

    recorder = {} if (chunking == "semantic" and conf.get("tree_build_diagnostics")) else None
    if chunking == "semantic":
        tqdm.write(
            f"Semantic chunking with {embed_model} "
            f"(threshold={conf.get('semantic_threshold')} {conf.get('semantic_threshold_type')}, "
            f"max_tokens={conf['max_tokens_per_chunk']}, short_chunk_tokens={conf.get('short_chunk_tokens')})..."
        )
    data.split_text(
        chunking=chunking,
        tokenizer=conf["tokenizer"],
        max_tokens=conf["max_tokens_per_chunk"],
        embed_model=embed_model,
        semantic_threshold=conf.get("semantic_threshold", 90),
        semantic_threshold_type=conf.get("semantic_threshold_type", "percentile"),
        distance_metric=conf.get("semantic_distance", "cosine"),
        short_chunk_tokens=conf.get("short_chunk_tokens", 0),
        merge_threshold=conf.get("semantic_merge_threshold"),
        embed_batch_size=conf.get("semantic_embed_batch_size", 64),
        distance_recorder=recorder,
    )
    if recorder is not None:
        report_chunking_diagnostics(recorder, conf)
    if getattr(data, "documents", None):
        data.finalize_chunks(title_prefix=bool(conf.get("enterprise_chunk_title_prefix", True)),
                             metadata_prefix=bool(conf.get("enterprise_chunk_metadata_prefix", False)))


def report_chunking_diagnostics(recorder: Dict, conf: Dict) -> None:
    """Print a summary of semantic chunking and dump the full record to save_dir."""
    documents = recorder.get("documents", [])
    chunks = [chunk for document in documents for chunk in document.get("chunks", [])]
    if not chunks:
        tqdm.write("Semantic chunking produced no chunks.")
        return

    tokens = [chunk["tokens"] for chunk in chunks]
    reasons: Dict[str, int] = {}
    for chunk in chunks:
        reason = chunk["split_info"]["reason"]
        reasons[reason] = reasons.get(reason, 0) + 1
    merged = sum(len(chunk["merged_from"]) for chunk in chunks)
    short_kept = sum(1 for chunk in chunks if chunk["tokens"] < conf.get("short_chunk_tokens", 0))
    triggering = [
        chunk["split_info"]["triggering_distance"] for chunk in chunks
        if chunk["split_info"]["triggering_distance"] is not None
    ]
    thresholds = [
        document["effective_threshold"] for document in documents
        if document.get("effective_threshold") is not None
    ]
    num_sentences = sum(document.get("num_sentences", 0) for document in documents)

    summary = (
        f"Semantic chunking stats: {len(documents)} documents, {num_sentences} sentences, "
        f"{len(chunks)} chunks; tokens/chunk min={min(tokens)} "
        f"avg={sum(tokens) / len(tokens):.1f} max={max(tokens)}; split reasons={reasons}; "
        f"short chunks merged={merged}, kept independent={short_kept}"
    )
    if thresholds:
        summary += f"; mean effective threshold={float(np.mean(thresholds)):.4f}"
    if triggering:
        summary += f"; mean semantic split distance={float(np.mean(triggering)):.4f}"
    tqdm.write(summary)

    if conf.get("save_dir") is not None:
        os.makedirs(conf["save_dir"], exist_ok=True)
        path = os.path.join(conf["save_dir"], f"chunking_diagnostics_{conf['dataset']}.json")
        with open(path, "w") as file:
            json.dump(recorder, file)
        tqdm.write(f'Chunking diagnostics saved to "{path}".')
