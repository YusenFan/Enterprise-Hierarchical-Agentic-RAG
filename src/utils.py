import logging
import re
import os
import json
import shutil
import string
import numpy as np
import tiktoken

from typing import Dict, Tuple, List, Optional, Sequence, Set
from pathlib import Path
from scipy import spatial
from tqdm import tqdm


logging.basicConfig(format="%(asctime)s - %(message)s", 
                    level=logging.INFO,
                    filename="./log/stdout.log",
                    filemode="a"
                    )


class Node:
    """
    Represents a node in the hierarchical tree structure.

    Leaf nodes (no children) are raw chunks: `document_index` / `chunk_index` are positions in the
    corpus, `document_id` references the source `Document` (see src/metadata) and `local_metadata`
    holds cheap chunk-level facts (speakers, section, ...). Abstract nodes (with children) carry
    `source_refs` ([{"document_id", "chunk_ids"}]), `source_document_ids` and `aggregated_metadata`,
    filled by `src.metadata.aggregate_tree_metadata`. The class-level defaults keep trees pickled
    before these fields existed loadable.
    """

    document_id: Optional[str] = None
    local_metadata: Optional[Dict] = None
    source_refs: Optional[List[Dict]] = None
    source_document_ids: Optional[List[str]] = None
    aggregated_metadata: Optional[Dict] = None

    def __init__(self, text: str, index: int, document_index: int, chunk_index: int, children: Set[int], embeddings: np.ndarray,
                 *, document_id: Optional[str] = None, local_metadata: Optional[Dict] = None) -> None:
        self.text: str = text
        self.index: int = index
        self.document_index: int = document_index
        self.chunk_index: int = chunk_index
        self.children: Set[int] = children
        self.embeddings: np.ndarray = embeddings
        if document_id is not None:
            self.document_id = document_id
        if local_metadata is not None:
            self.local_metadata = local_metadata

    @property
    def is_leaf(self) -> bool:
        return not self.children


class Tree:
    """
    Represents the entire hierarchical tree structure.
    `documents` (optional) maps document_id -> Document for provenance / context headers.
    """

    documents: Optional[Dict] = None

    def __init__(
        self, all_nodes, root_nodes, leaf_nodes, layer_to_node_indices, documents: Optional[Dict] = None
    ) -> None:
        self.all_nodes: Dict[int, Node] = all_nodes
        self.root_nodes: Dict[int, Node] = root_nodes
        self.leaf_nodes: Dict[int, Node] = leaf_nodes
        self.num_layers: int = max(layer_to_node_indices.keys())
        # self.num_layers = num_layers
        self.layer_to_node_indices: Dict[int, List[int]] = layer_to_node_indices
        if documents is not None:
            self.documents = documents


def repair_node_indices(tree: "Tree") -> int:
    """
    Make every node's `.index` equal to its key in `tree.all_nodes`. Trees built with preset
    chunks used to keep the *document position* as the index of layer-1 nodes, so any code that
    reads `node.index` (retrieval provenance, document credit) pointed at an unrelated leaf.
    Returns the number of repaired nodes. Idempotent; safe on every builder's output.
    """
    repaired = 0
    for key, node in tree.all_nodes.items():
        if node.index != key:
            node.index = key
            repaired += 1
    if repaired:
        logging.warning(f"repair_node_indices: {repaired} nodes had a stale .index (fixed in memory).")
    return repaired


def reverse_mapping(layer_to_node_indices: Dict[int, List[int]]) -> Dict[int, int]:
    node_to_layer = {}
    for layer, nodes in layer_to_node_indices.items():
        for node in nodes:
            node_to_layer[node] = layer
    return node_to_layer


def _resolve_tokenizer(tokenizer) -> tiktoken.Encoding:
    """Accepts a tiktoken encoding name or an Encoding instance and returns the Encoding."""
    if tokenizer is None:
        raise ValueError("There is no tokenizer. ")
    if isinstance(tokenizer, str):
        return tiktoken.get_encoding(tokenizer)
    return tokenizer


def split_text(
    text: str, tokenizer: str | tiktoken.Encoding, max_tokens: int, overlap: int = 0
) -> List[str]:
    """
    Splits the input text into smaller chunks based on the tokenizer and maximum allowed tokens.
    
    Args:
        text (str): The text to be split.
        tokenizer (CustomTokenizer): The tokenizer to be used for splitting the text.
        max_tokens (int): The maximum allowed tokens.
        overlap (int, optional): The number of overlapping tokens between chunks. Defaults to 0.
    
    Returns:
        List[str]: A list of text chunks.
    """
    tokenizer = _resolve_tokenizer(tokenizer)

    # Split the text into sentences using multiple delimiters
    delimiters = [".", "!", "?", "\n"]
    regex_pattern = "|".join(map(re.escape, delimiters))
    sentences = re.split(regex_pattern, text)
    
    # Calculate the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence, token_count in zip(sentences, n_tokens):
        # If the sentence is empty or consists only of whitespace, skip it
        if not sentence.strip():
            continue
        
        # If the sentence is too long, split it into smaller parts
        if token_count > max_tokens:
            sub_sentences = re.split(r"[,;:]", sentence)
            
            # there is no need to keep empty os only-spaced strings
            # since spaces will be inserted in the beginning of the full string
            # and in between the string in the sub_chunk list
            filtered_sub_sentences = [sub.strip() for sub in sub_sentences if sub.strip() != ""]
            sub_token_counts = [len(tokenizer.encode(" " + sub_sentence)) 
                                for sub_sentence in filtered_sub_sentences]
            
            sub_chunk = []
            sub_length = 0
            
            for sub_sentence, sub_token_count in zip(filtered_sub_sentences, sub_token_counts):
                if sub_length + sub_token_count > max_tokens:
                    
                    # if the phrase does not have sub_sentences, it would create an empty chunk
                    # this big phrase would be added anyways in the next chunk append
                    if sub_chunk:
                        chunks.append(" ".join(sub_chunk))
                        sub_chunk = sub_chunk[-overlap:] if overlap > 0 else []
                        sub_length = sum(sub_token_counts[max(0, len(sub_chunk) - overlap):len(sub_chunk)])
                
                sub_chunk.append(sub_sentence)
                sub_length += sub_token_count
            
            if sub_chunk:
                chunks.append(" ".join(sub_chunk))
        
        # If adding the sentence to the current chunk exceeds the max tokens, start a new chunk
        elif current_length + token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_length = sum(n_tokens[max(0, len(current_chunk) - overlap):len(current_chunk)])
            current_chunk.append(sentence)
            current_length += token_count
        
        # Otherwise, add the sentence to the current chunk
        else:
            current_chunk.append(sentence)
            current_length += token_count
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


DEFAULT_SENTENCE_DELIMITERS = (".", "!", "?", "\n")


def split_sentences(
    text: str, sentence_delimiters: Sequence[str] = DEFAULT_SENTENCE_DELIMITERS
) -> List[str]:
    """
    Splits text into sentences, keeping each delimiter attached to its sentence.

    Punctuation delimiters only end a sentence when they are followed by whitespace
    (or the end of the text), so tokens like "3.14" or "www.example.com" are not cut apart.
    Line breaks always end a sentence.

    Args:
        text (str): The text to be split.
        sentence_delimiters (Sequence[str]): Characters that end a sentence.

    Returns:
        List[str]: Non-empty, stripped sentences in document order.
    """
    if not text or not text.strip():
        return []

    punctuation = [d for d in sentence_delimiters if d not in ("\n", "\r")]
    patterns = []
    if punctuation:
        char_class = "".join(re.escape(d) for d in punctuation)
        patterns.append(rf"(?<=[{char_class}])\s+")
    if any(d in ("\n", "\r") for d in sentence_delimiters):
        patterns.append(r"[\r\n]+")
    if not patterns:
        return [text.strip()]

    pieces = re.split("|".join(patterns), text)
    return [piece.strip() for piece in pieces if piece and piece.strip()]


def _consecutive_distances(embeddings: np.ndarray, distance_metric: str = "cosine") -> np.ndarray:
    """Distances between each pair of consecutive rows of `embeddings` (vectorised)."""
    if len(embeddings) < 2:
        return np.zeros(0, dtype=float)
    a, b = embeddings[:-1], embeddings[1:]
    if distance_metric == "cosine":
        norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            cosine = np.sum(a * b, axis=1) / norms
        cosine = np.where(norms > 0, cosine, 0.0)
        return 1.0 - np.clip(cosine, -1.0, 1.0)
    if distance_metric == "L1":
        return np.sum(np.abs(a - b), axis=1)
    if distance_metric == "L2":
        return np.linalg.norm(a - b, axis=1)
    if distance_metric == "Linf":
        return np.max(np.abs(a - b), axis=1)
    raise ValueError(
        f"Unsupported distance metric '{distance_metric}'. "
        "Supported metrics are: ['cosine', 'L1', 'L2', 'Linf']"
    )


def _split_long_sentence(sentence: str, tokenizer: tiktoken.Encoding, max_tokens: int) -> List[str]:
    """
    Splits a sentence longer than `max_tokens` with the static splitter ("[,;:]" fallback);
    any piece that still exceeds `max_tokens` (no delimiters at all, e.g. an infobox or a
    table row) is hard-cut into token windows so the bound always holds.
    """
    pieces = []
    for piece in split_text(sentence, tokenizer, max_tokens):
        token_ids = tokenizer.encode(" " + piece)
        if len(token_ids) <= max_tokens:
            pieces.append(piece)
            continue
        for start in range(0, len(token_ids), max_tokens):
            window = tokenizer.decode(token_ids[start: start + max_tokens]).strip()
            if window:
                pieces.append(window)
    return pieces


def _new_chunk(text: str, tokens: int, embedding: np.ndarray, reason: str = None) -> Dict:
    return {
        "texts": [text],
        "tokens": int(tokens),
        "emb_sum": np.asarray(embedding, dtype=float).copy(),
        "n_sents": 1,
        "distances": [],
        "reason": reason,
        "triggering_distance": None,
        "merged_from": [],
    }


def _chunk_embedding(chunk: Dict) -> np.ndarray:
    """Chunk embedding = mean of its sentence embeddings."""
    return chunk["emb_sum"] / max(chunk["n_sents"], 1)


def _merge_short_chunks(
    chunks: List[Dict],
    short_chunk_tokens: int,
    merge_threshold: Optional[float],
    max_tokens: int,
    distance_metric: str = "cosine",
) -> List[Dict]:
    """
    Post-pass over provisional semantic chunks. A chunk shorter than `short_chunk_tokens`
    is compared with its left and right neighbours; it is merged into the more similar one
    only if it is NOT semantically independent (distance <= merge_threshold) and the merged
    chunk still fits in `max_tokens`. Independent short chunks are kept as they are.
    """
    if short_chunk_tokens <= 0 or merge_threshold is None or len(chunks) < 2:
        return chunks

    chunks = list(chunks)
    merged = True
    while merged and len(chunks) > 1:
        merged = False
        i = 0
        while i < len(chunks) and len(chunks) > 1:
            chunk = chunks[i]
            if chunk["tokens"] >= short_chunk_tokens:
                i += 1
                continue

            candidates = []
            if i > 0:
                candidates.append(("left", i - 1))
            if i < len(chunks) - 1:
                candidates.append(("right", i + 1))
            neighbour_embeddings = [_chunk_embedding(chunks[j]) for _, j in candidates]
            neighbour_distances = distances_from_embeddings(
                _chunk_embedding(chunk), neighbour_embeddings, distance_metric
            )
            k = int(np.argmin(neighbour_distances))
            side, j = candidates[k]
            distance = float(neighbour_distances[k])
            neighbour = chunks[j]

            if distance <= merge_threshold and chunk["tokens"] + neighbour["tokens"] <= max_tokens:
                left, right = (neighbour, chunk) if side == "left" else (chunk, neighbour)
                merged_chunk = {
                    "texts": left["texts"] + right["texts"],
                    "tokens": left["tokens"] + right["tokens"],
                    "emb_sum": left["emb_sum"] + right["emb_sum"],
                    "n_sents": left["n_sents"] + right["n_sents"],
                    "distances": left["distances"] + right["distances"],
                    "reason": right["reason"],
                    "triggering_distance": right["triggering_distance"],
                    "merged_from": left["merged_from"] + right["merged_from"] + [{
                        "text": " ".join(chunk["texts"]),
                        "tokens": chunk["tokens"],
                        "merged_into": side,
                        "distance": distance,
                    }],
                }
                first = min(i, j)
                chunks[first] = merged_chunk
                del chunks[max(i, j)]
                merged = True
                i = first
            else:
                i += 1
    return chunks


def chunk_sentences_semantic(
    sentences: List[str],
    sentence_embeddings,
    tokenizer: str | tiktoken.Encoding,
    max_tokens: int,
    semantic_threshold: float,
    semantic_threshold_type: str = "percentile",
    distance_metric: str = "cosine",
    short_chunk_tokens: int = 0,
    merge_threshold: Optional[float] = None,
    distance_recorder: Optional[Dict] = None,
) -> List[str]:
    """
    Groups consecutive sentences into chunks using the semantic distance between
    neighbouring sentences and a maximum token count per chunk.

    Args:
        sentences (List[str]): Sentences in document order (see `split_sentences`).
        sentence_embeddings: One embedding per sentence, shape (len(sentences), dim).
        tokenizer: tiktoken encoding name or Encoding used to count tokens.
        max_tokens (int): Maximum token count of one chunk.
        semantic_threshold (float): Split threshold, see `semantic_threshold_type`.
        semantic_threshold_type (str): "percentile" splits where the distance exceeds the
            `semantic_threshold`-th percentile of this document's consecutive-sentence
            distances; "absolute" splits where the distance > `semantic_threshold`.
        distance_metric (str): ("cosine", "L1", "L2", "Linf").
        short_chunk_tokens (int): Chunks shorter than this are candidates for merging into
            their more similar neighbour (see `_merge_short_chunks`). 0 disables the post-pass.
        merge_threshold (float | None): Distance below which a short chunk counts as not
            semantically independent. None reuses the effective split threshold.
        distance_recorder (Dict | None): If given, filled with per-chunk diagnostics.

    Returns:
        List[str]: Text chunks in document order.
    """
    if not sentences:
        return []
    tokenizer = _resolve_tokenizer(tokenizer)

    embeddings = np.asarray(sentence_embeddings, dtype=float)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    if len(embeddings) != len(sentences):
        raise ValueError(
            f"Got {len(embeddings)} sentence embeddings for {len(sentences)} sentences."
        )

    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    distances = _consecutive_distances(embeddings, distance_metric)

    if semantic_threshold_type == "percentile":
        if not 0 <= semantic_threshold <= 100:
            raise ValueError("A percentile semantic_threshold must be within [0, 100].")
        threshold = float(np.percentile(distances, semantic_threshold)) if len(distances) else None
    elif semantic_threshold_type == "absolute":
        threshold = float(semantic_threshold)
    else:
        raise ValueError(
            f"Unsupported semantic_threshold_type '{semantic_threshold_type}'. "
            "Expected 'percentile' or 'absolute'."
        )
    if merge_threshold is None:
        merge_threshold = threshold

    chunks: List[Dict] = []
    current: Optional[Dict] = None

    def close(reason: str, triggering_distance: Optional[float] = None) -> None:
        nonlocal current
        if current is not None:
            current["reason"] = reason
            current["triggering_distance"] = triggering_distance
            chunks.append(current)
            current = None

    for i, sentence in enumerate(sentences):
        tokens = n_tokens[i]

        # A single sentence longer than max_tokens: close the running chunk and
        # sub-split the sentence with the static splitter ("[,;:]" fallback).
        if tokens > max_tokens:
            close("followed_by_long_sentence")
            for piece in _split_long_sentence(sentence, tokenizer, max_tokens):
                piece_tokens = len(tokenizer.encode(" " + piece))
                chunks.append(_new_chunk(piece, piece_tokens, embeddings[i], reason="single_long_sentence"))
            continue

        if current is None:
            current = _new_chunk(sentence, tokens, embeddings[i])
            continue

        distance = float(distances[i - 1])
        split_semantic = threshold is not None and distance > threshold
        split_length = current["tokens"] + tokens > max_tokens
        if split_semantic or split_length:
            close("semantic" if split_semantic else "length", distance if split_semantic else None)
            current = _new_chunk(sentence, tokens, embeddings[i])
        else:
            current["texts"].append(sentence)
            current["tokens"] += tokens
            current["emb_sum"] += embeddings[i]
            current["n_sents"] += 1
            current["distances"].append(distance)
    close("end_of_text")

    chunks = _merge_short_chunks(chunks, short_chunk_tokens, merge_threshold, max_tokens, distance_metric)

    if distance_recorder is not None:
        distance_recorder["num_sentences"] = len(sentences)
        distance_recorder["effective_threshold"] = threshold
        distance_recorder["merge_threshold"] = merge_threshold
        distance_recorder["chunks"] = [
            {
                "chunk_text": " ".join(chunk["texts"]),
                "tokens": int(chunk["tokens"]),
                "distances_between_sentences": [float(d) for d in chunk["distances"]],
                "average_distance": float(np.mean(chunk["distances"])) if chunk["distances"] else 0.0,
                "split_info": {
                    "reason": chunk["reason"],
                    "triggering_distance": chunk["triggering_distance"],
                },
                "merged_from": chunk["merged_from"],
            }
            for chunk in chunks
        ]

    return [" ".join(chunk["texts"]) for chunk in chunks if chunk["texts"]]


def split_text_semantic(
    text: str,
    tokenizer: str | tiktoken.Encoding,
    max_tokens: int,
    embedding_function,
    semantic_threshold: float,
    semantic_threshold_type: str = "percentile",
    distance_metric: str = "cosine",
    sentence_delimiters: Sequence[str] = DEFAULT_SENTENCE_DELIMITERS,
    short_chunk_tokens: int = 0,
    merge_threshold: Optional[float] = None,
    distance_recorder: Optional[Dict] = None,
) -> List[str]:
    """
    Splits text into chunks by the semantic distance between consecutive sentences,
    capped by `max_tokens` per chunk. Falls back to the static `split_text` if the
    embedding call fails.

    Args:
        embedding_function: Callable mapping List[str] -> embeddings of shape (n, dim).
        Other arguments: see `chunk_sentences_semantic` and `split_sentences`.
    """
    if not text or not text.strip():
        return []

    sentences = split_sentences(text, sentence_delimiters)
    if not sentences:
        return []

    try:
        logging.info(f"Embedding {len(sentences)} sentences for semantic chunking...")
        sentence_embeddings = np.asarray(embedding_function(sentences), dtype=float)
        if sentence_embeddings.ndim == 1 and len(sentences) == 1:
            sentence_embeddings = sentence_embeddings.reshape(1, -1)
        if len(sentence_embeddings) != len(sentences):
            raise ValueError(
                f"Got {len(sentence_embeddings)} embeddings for {len(sentences)} sentences."
            )
    except Exception as e:
        logging.error(f"Sentence embedding failed during semantic chunking: {e}")
        logging.warning("Falling back to static splitting for this text.")
        return split_text(text, tokenizer, max_tokens)

    return chunk_sentences_semantic(
        sentences,
        sentence_embeddings,
        tokenizer,
        max_tokens,
        semantic_threshold,
        semantic_threshold_type=semantic_threshold_type,
        distance_metric=distance_metric,
        short_chunk_tokens=short_chunk_tokens,
        merge_threshold=merge_threshold,
        distance_recorder=distance_recorder,
    )


def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric: str = "cosine",
) -> List[float]:
    """
    Calculates the distances between a query embedding and a list of embeddings.

    Args:
        query_embedding (List[float]): The query embedding.
        embeddings (List[List[float]]): A list of embeddings to compare against the query embedding.
        distance_metric (str, optional): The distance metric to use for calculation. Defaults to 'cosine'.

    Returns:
        List[float]: The calculated distances between the query embedding and the list of embeddings.
    """
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }

    if distance_metric not in distance_metrics:
        raise ValueError(
            f"Unsupported distance metric '{distance_metric}'. Supported metrics are: {list(distance_metrics.keys())}"
        )

    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]

    return distances


def get_node_list(node_dict: Dict[int, Node]) -> List[Node]:
    """
    Converts a dictionary of node indices to a sorted list of nodes.

    Args:
        node_dict (Dict[int, Node]): Dictionary of node indices to nodes.

    Returns:
        List[Node]: Sorted list of nodes.
    """
    indices = sorted(node_dict.keys())
    node_list = [node_dict[index] for index in indices]
    return node_list


def get_embeddings(node_list: List[Node]) -> List:
    """
    Extracts the embeddings of nodes from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.
        embedding_model (str): The name of the embedding model to be used.

    Returns:
        List: List of node embeddings.
    """
    return [node.embeddings for node in node_list]


def get_children(node_list: List[Node]) -> List[Set[int]]:
    """
    Extracts the children of nodes from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.

    Returns:
        List[Set[int]]: List of sets of node children indices.
    """
    return [node.children for node in node_list]


def get_text(node_list: List[Node]) -> str:
    """
    Generates a single text string by concatenating the text from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.

    Returns:
        str: Concatenated text.
    """
    text = ""
    for node in node_list:
        text += f"{' '.join(node.text.splitlines())}"
        text += "\n\n"
    return text


def get_text_list(node_list: List[Node]) -> List[str]:
    """
    Generates a text string list from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.

    Returns:
        List: List of text.
    """
    text_list = []
    for node in node_list:
        text_list.append(node.text)
    return text_list


def get_token_length(text: str | List[str]) -> Dict:

    if isinstance(text, str):
        text = [text]
    
    tokens = []

    for t in text:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        # Split the text into sentences using multiple delimiters
        delimiters = [".", "!", "?", "\n"]
        regex_pattern = "|".join(map(re.escape, delimiters))
        sentences = re.split(regex_pattern, t)
        
        # Calculate the number of tokens for each sentence
        tokens.append(sum([len(tokenizer.encode(" " + sentence)) for sentence in sentences]))
    
    return {
        "total_token": sum(tokens), 
        "avg_token": float(sum(tokens)) / len(text),
        "max_token": max(tokens),
        "min_token": min(tokens),
    }


def is_bucketed_tree(conf: Dict) -> bool:
    return conf.get("bucket_size") is not None


def sanitize_save_name(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name


def dataset_tag(conf: Dict) -> str:
    """
    Dataset part of index / result file names. EnterpriseRAG-Bench subsets are tagged with their
    size and seed so a tree and a BM25 index built on different subsets can never be paired.
    """
    dataset = conf["dataset"]
    if dataset == "enterprise_rag":
        size = conf.get("enterprise_subset_size")
        seed = conf.get("enterprise_subset_seed", 42)
        return "enterprise_rag_full" if size is None else f"enterprise_rag_n{size}_s{seed}"
    return dataset


def get_tree_save_name(conf: Dict) -> str:
    prefix = (
        f'{dataset_tag(conf)}_{conf["embed_name"].replace("/", "_")}'
        f'_{str(conf["abs_name"]).replace("/", "_")}_{conf["abstract_type"]}'
    )
    if conf.get("chunking") == "semantic":
        prefix = f"{prefix}_semantic"
    if conf.get("enterprise_chunk_metadata_prefix"):
        prefix = f"{prefix}_metatext"
    tree_builder = conf.get("tree_builder", "exact")

    if is_bucketed_tree(conf):
        return sanitize_save_name(f"{prefix}_{tree_builder}_bucketed_tree")
    if tree_builder != "exact":
        return sanitize_save_name(f"{prefix}_{tree_builder}_tree.pkl")
    return sanitize_save_name(f"{prefix}_tree.pkl")


def get_sparse_save_name(conf: Dict) -> str:
    """Directory name of the BM25 index. Tagged like the tree so both always match."""
    name = f"bm25_{dataset_tag(conf)}"
    if conf.get("chunking") == "semantic":
        name = f"{name}_semantic"
    if conf.get("enterprise_chunk_metadata_prefix"):
        name = f"{name}_metatext"
    return name


def remove_tree_target(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def check_single_tree(tree) -> bool:
    if not hasattr(tree, "root_nodes") or len(tree.root_nodes) != 1:
        return False

    root_node_index = next(iter(tree.root_nodes.keys()))
    visited = set()
    stack = [root_node_index]

    while stack:
        node_index = stack.pop()
        if node_index in visited:
            continue
        visited.add(node_index)
        stack.extend(list(tree.all_nodes[node_index].children))

    return len(visited) == len(tree.all_nodes)


def print_tree_check(tree) -> None:
    if isinstance(tree, list):
        is_connected = len(tree) > 0 and all(check_single_tree(t) for t in tree)
        if is_connected:
            tqdm.write(f"Tree check passed: all {len(tree)} trees are connected.")
        else:
            tqdm.write("Tree check failed.")
        return

    if check_single_tree(tree):
        tqdm.write(f"Tree check passed: all {len(tree.all_nodes)} nodes are in one tree.")
    else:
        tqdm.write("Tree check failed.")


def prototype_embeddings(embeddings, method='average'):
    if method == 'average':
        return np.mean(embeddings, axis=0)
    elif method == 'tf-idf':
        return NotImplementedError
    else:
        return NotImplementedError


def normalize_answer(answer: str) -> str:
    """
    Normalize a given string by applying the following transformations:
    1. Convert the string to lowercase.
    2. Remove punctuation characters.
    3. Remove the articles "a", "an", and "the".
    4. Normalize whitespace by collapsing multiple spaces into one.

    Args:
        answer (str): The input string to be normalized.

    Returns:
        str: The normalized string.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(answer))))


def parse_response(response, verbose=False):
    if isinstance(response, str):
        thought, action, info = response.rpartition('Retrieve: ')
        if action != 'Retrieve: ':
            thought, action, info = response.rpartition('Answer: ')
            if action != 'Answer: ':
                if verbose:
                    tqdm.write(f"Warning: LLM output not as intended: \n{response}")
                return "", "answer", "Error"
        
        return thought.rstrip('\n'), action.rstrip(' :').lower(), info.rstrip(' .')
    else: # dict
        if response.retrieve is not None and response.answer is None:
            return response.thought, "retrieve", response.retrieve
        elif response.retrieve is None and response.answer is not None:
            return response.thought, "answer", response.answer
        else:
            raise ValueError(f"response.answer: {response.answer}\nresponse.retrieve: {response.retrieve}")
            

def rrf(docs: List[List[str]], top_k: int = 5, k: int = 60) -> Tuple[List[str], List[float]]:
    node_scoring_sheet = {}
    for passages in docs:
        for rank, passage in enumerate(passages, 1):
            node_scoring_sheet.setdefault(passage, 0)
            node_scoring_sheet[passage] += 1 / (rank + k)
    
    node_scoring_sheet = dict(sorted(node_scoring_sheet.items(), key=lambda x: x[1], reverse=True))
    return list(node_scoring_sheet.keys())[:top_k], list(node_scoring_sheet.values())[:top_k]


def result_stem(conf: Dict) -> str:
    """
    Base name of result / log / judge-cache files: "<config>" or "<config>_<run_tag>" so several
    retrieval variants of one config can coexist (used by qa.py, main.py, eval.py, run_arms.py).
    """
    stem = str(conf["config"])
    run_tag = conf.get("run_tag")
    if run_tag:
        stem = f"{stem}_{sanitize_save_name(str(run_tag)).replace(' ', '_')}"
    return stem


def save_answers(conf: Dict, results: Dict, ans_path: Path, single_query: bool = False, verbose: bool = False) -> None:
    if not single_query:
        results["conf"] = repr(conf)
    if not os.path.exists(ans_path):
        os.makedirs(ans_path)
    file_name = f"{result_stem(conf)}_query.json" if single_query else f"{result_stem(conf)}.json"
    ans_file = os.path.join(ans_path, file_name)
    with open(ans_file, "w") as f:
        json.dump(results, f)
    logging.info(f"QA results successfully saved to {ans_file}!")
    if verbose:
        tqdm.write(f"Question answering completed! Answer file saved to \"{ans_file}\".")


def load_answers(conf: Dict, single_query: bool = False) -> None | Tuple[List[str], List[List[str]]] | Dict:
    if conf["save_dir"] is None and single_query:
        return {}

    file_name = f"{result_stem(conf)}_query.json" if single_query else f"{result_stem(conf)}.json"
    ans_file = os.path.join(conf["save_dir"], "results", file_name)
    if not os.path.exists(ans_file):
        logging.info("No answer file detected. Running from scratch.")
        return {} if single_query else None
    if conf["force_qa_from_scratch"]:
        logging.info("\"force_qa_from_scratch\" is on. Running from scratch.")
        if not single_query:
            os.remove(ans_file)
            return None
        return {}
    with open(ans_file, "r") as f:
        results = json.load(f)
    if single_query:
        if not isinstance(results, dict):
            logging.info("Invalid result file.")
            return {}
        logging.info(f"QA results successfully loaded from {ans_file}!")
        return results
    if not {"conf", "answers"}.issubset(results.keys()):
        logging.info("Invalid result file.")
        return None
    logging.info(f"QA results successfully loaded from {ans_file}!")
    return results
