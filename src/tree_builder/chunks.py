import glob
import json
import os
import pickle

from typing import Dict

from ..metadata.document import Document
from ..utils import Node, Tree

METADATA_FIELDS = ("document_id", "local_metadata", "source_refs", "source_document_ids", "aggregated_metadata")


def _serialize_node(node: Node) -> Dict:
    data = {
        "text": node.text,
        "index": node.index,
        "document_index": node.document_index,
        "chunk_index": node.chunk_index,
        "children": sorted(node.children),
        "embeddings": node.embeddings,
    }
    for field in METADATA_FIELDS:
        value = getattr(node, field, None)
        if value is not None:
            data[field] = value
    return data


def _deserialize_node(data: Dict) -> Node:
    node = Node(
        text=data["text"],
        index=data["index"],
        document_index=data["document_index"],
        chunk_index=data["chunk_index"],
        children=set(data["children"]),
        embeddings=data["embeddings"],
    )
    for field in METADATA_FIELDS:
        if data.get(field) is not None:
            setattr(node, field, data[field])
    return node


def save_tree_chunks(tree: Tree, path: str, chunk_size: int = 200000) -> None:
    os.makedirs(path, exist_ok=True)

    for stale_chunk in glob.glob(os.path.join(path, "nodes_*.pkl")):
        os.remove(stale_chunk)
    manifest_path = os.path.join(path, "manifest.json")
    if os.path.exists(manifest_path):
        os.remove(manifest_path)

    items = sorted(tree.all_nodes.items(), key=lambda x: x[0])
    chunk_files = []
    for chunk_idx, start in enumerate(range(0, len(items), chunk_size)):
        chunk_file = f"nodes_{chunk_idx:05d}.pkl"
        chunk_path = os.path.join(path, chunk_file)
        node_chunk = {
            idx: _serialize_node(node)
            for idx, node in items[start: start + chunk_size]
        }
        with open(chunk_path, "wb") as file:
            pickle.dump(node_chunk, file)
        chunk_files.append(chunk_file)

    documents = getattr(tree, "documents", None)
    if documents:
        with open(os.path.join(path, "documents.json"), "w", encoding="utf-8") as file:
            json.dump({doc_id: doc.to_dict() for doc_id, doc in documents.items()}, file)

    manifest = {
        "format": "bucketed-tree-v2",
        "chunk_size": chunk_size,
        "chunk_files": chunk_files,
        "root_node_indices": sorted(tree.root_nodes.keys()),
        "leaf_node_indices": sorted(tree.leaf_nodes.keys()),
        "layer_to_node_indices": tree.layer_to_node_indices,
    }
    with open(os.path.join(path, "manifest.json"), "w") as file:
        json.dump(manifest, file)


def load_tree_chunks(path: str) -> Tree:
    manifest_path = os.path.join(path, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as file:
            manifest = json.load(file)
        chunk_files = manifest.get("chunk_files", [])
        root_indices = manifest["root_node_indices"]
        leaf_indices = manifest["leaf_node_indices"]
        layer_to_node_indices = {
            int(k): v for k, v in manifest["layer_to_node_indices"].items()
        }
    else:
        chunk_files = [
            os.path.basename(name)
            for name in sorted(glob.glob(os.path.join(path, "nodes_*.pkl")))
        ]
        root_indices = []
        leaf_indices = []
        layer_to_node_indices = {}

    all_nodes = {}
    for chunk_file in chunk_files:
        chunk_path = os.path.join(path, chunk_file)
        with open(chunk_path, "rb") as file:
            node_chunk = pickle.load(file)
        for idx, node_data in node_chunk.items():
            all_nodes[idx] = _deserialize_node(node_data)

    if not root_indices:
        raise ValueError(f"No manifest metadata found in tree directory: {path}")

    root_nodes = {idx: all_nodes[idx] for idx in root_indices}
    leaf_nodes = {idx: all_nodes[idx] for idx in leaf_indices}

    documents = None
    documents_path = os.path.join(path, "documents.json")
    if os.path.exists(documents_path):
        with open(documents_path, "r", encoding="utf-8") as file:
            documents = {doc_id: Document.from_dict(d) for doc_id, d in json.load(file).items()}

    return Tree(
        all_nodes=all_nodes,
        root_nodes=root_nodes,
        leaf_nodes=leaf_nodes,
        layer_to_node_indices=layer_to_node_indices,
        documents=documents,
    )
