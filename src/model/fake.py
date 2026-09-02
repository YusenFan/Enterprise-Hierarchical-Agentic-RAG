"""
Offline stand-ins for the embedding / abstraction / QA backends.

They exist so the whole index -> qa -> eval pipeline (and the test-suite) can run
without network access or GPU: "hash:bow" embeds text with a signed bag-of-words hash,
"fake:abstract" truncates the input, "fake:qa" echoes the question. They are selected
through the usual "<platform>:<model>" names (see src/model/factory.py).
"""
import hashlib
import re
from typing import List

import numpy as np

from .abstract import BaseAbstractModel
from .embed import BaseEmbeddingModel
from .qa import BaseQAModel

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashEmbeddingModel(BaseEmbeddingModel):
    """Deterministic signed bag-of-words hashing embedding (no model download)."""

    def __init__(self, model_name: str = "bow", cache_dir=None, dim: int = 256, **kwargs) -> None:
        self.model_name = f"hash:{model_name}"
        self.dim = int(dim)

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in _TOKEN_RE.findall(str(text).lower()):
            digest = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            sign = 1.0 if (digest >> 64) & 1 == 0 else -1.0
            vec[digest % self.dim] += sign
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            vec[0] = 1.0
            norm = 1.0
        return vec / norm

    def embed(self, text) -> np.ndarray:
        if isinstance(text, (list, tuple)):
            return self.embed_batch(list(text))
        return self._embed_one(text)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.stack([self._embed_one(t) for t in texts])


class FakeAbstractModel(BaseAbstractModel):
    """Returns the first `max_abs_length` words of the context."""

    def __init__(self, model_name: str = "abstract", cache_dir=None, **kwargs) -> None:
        self.model_name = f"fake:{model_name}"

    def abstract(self, context, keyword=False, max_abs_length: int = 100, leaf: bool = True) -> str:
        words = str(context).split()
        return " ".join(words[: max(int(max_abs_length), 1)])


class FakeQAModel(BaseQAModel):
    """Echoes a short answer derived from the user question (agent-protocol compatible)."""

    def __init__(self, model_name: str = "qa", cache_dir=None, **kwargs) -> None:
        self.model_name = f"fake:{model_name}"

    @staticmethod
    def _last_user_content(question) -> str:
        if isinstance(question, str):
            return question
        for message in reversed(list(question)):
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content", ""))
        return ""

    def qa(self, question, max_tokens: int = 1000, stream: bool = False, **kwargs) -> str:
        content = self._last_user_content(question)
        marker = "User question:"
        if marker in content:
            content = content.rsplit(marker, maxsplit=1)[1]
        words = content.strip().split()
        return "Thought: fake model.\nAnswer: " + " ".join(words[:20])
