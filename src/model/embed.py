import os
import logging
import threading

from abc import ABC, abstractmethod
from typing import List
from tqdm import tqdm

import torch
import ollama
import numpy as np
import transformers
from torch.nn.functional import normalize
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential

logging.basicConfig(format="%(asctime)s - %(message)s", 
                    level=logging.INFO,
                    filename="./log/stdout.log",
                    filemode="a"
                    )


class BaseEmbeddingModel(ABC):
    model_name: str
    # Lazy model loading can be triggered from several threads at once
    # (leaf nodes are embedded in a ThreadPoolExecutor); serialise it.
    _load_lock = threading.Lock()

    @abstractmethod
    def embed(self, text) -> np.ndarray:
        pass

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a list of texts and returns an array of shape (len(texts), dim).
        Backends that support batched inputs override this; the default embeds one text at a time.
        """
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not texts:
            return np.empty((0, 0), dtype=float)
        return np.asarray(
            [np.asarray(self.embed(text), dtype=float).reshape(-1) for text in texts]
        )

    def __repr__(self):
        return self.model_name


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="text-embedding-3-small", cache_dir=None, **kwargs):
        """
        Embeddings from an OpenAI-compatible endpoint. Reads OPENAI_API_KEY and, optionally,
        OPENAI_BASE_URL from the environment (conf/__init__.py loads them from ".env").
        Leave OPENAI_BASE_URL empty for the official endpoint (https://api.openai.com/v1).

        Args:
            model_name (str): text-embedding-3-small, text-embedding-3-large,
                              text-embedding-ada-002 (official ids carry no "openai/" prefix)
            dimensions (int, optional): output size for text-embedding-3-* models.
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model_kwargs = kwargs
        self.dimensions = self.model_kwargs.pop("dimensions", None)

        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                'OPENAI_API_KEY is not set. Put OPENAI_API_KEY (and optionally OPENAI_BASE_URL) '
                'in ".env" at the repo root (see ".env.example") or export them in your shell.'
            )
        # An empty OPENAI_BASE_URL must mean the official endpoint; the SDK would otherwise use "".
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or "https://api.openai.com/v1"
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def _request_kwargs(self):
        request_kwargs = {"model": self.model_name}
        if self.dimensions is not None:
            request_kwargs["dimensions"] = self.dimensions
        return request_kwargs

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def embed(self, text):
        if not isinstance(text, str):
            return self.embed_batch(text)
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(input=[text], **self._request_kwargs())
        return response.data[0].embedding

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        texts = [text.replace("\n", " ") for text in texts]
        if not texts:
            return np.empty((0, 0), dtype=float)
        response = self.client.embeddings.create(input=texts, **self._request_kwargs())
        data = sorted(response.data, key=lambda item: item.index)
        return np.asarray([item.embedding for item in data], dtype=float)


class SentenceTransformersEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1", cache_dir=None, **kwargs):
        """
        Args:
            model_name (str): sentence-transformers/multi-qa-mpnet-base-cos-v1, 
                              sentence-transformers/all-mpnet-base-v2,
                              nvidia/NV-Embed-v2,
                              Qwen/Qwen3-Embedding-0.6B,
                              Qwen/Qwen3-Embedding-8B,
        """
        self.model_name = model_name
        self.model = None
        self.cache_dir = cache_dir

    def load_model(self):
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:
                return
            if self.model_name in ["nvidia/NV-Embed-v2"]:
                assert transformers.__version__ <= "4.46.0", "Use old transformers package (<= 4.46.0)."
                self.model = SentenceTransformer(self.model_name, 
                                                 trust_remote_code=True, 
                                                 cache_folder=self.cache_dir,
                                                 config_kwargs={'use_cache': False},
                                                 )
                self.model.max_seq_length = 32768
                self.model.tokenizer.padding_side="right"
            else:
                self.model = SentenceTransformer(self.model_name)

    def add_eos(self, input_examples):
        input_examples = [input_example + self.model.tokenizer.eos_token for input_example in input_examples]
        return input_examples

    def embed(self, text, **kwargs):
        self.load_model()
            
        if self.model_name in ("nvidia/NV-Embed-v2",):
            text = self.add_eos(text)

            kwargs.setdefault("batch_size", 8)
            kwargs.setdefault("prompt", "")
            kwargs.setdefault("normalize_embeddings", True)
        elif self.model_name in ("Qwen/Qwen3-Embedding-8B",):
            kwargs.setdefault("batch_size", 32)
            
        return self.model.encode(text, show_progress_bar=False, **kwargs)

    def embed_batch(self, texts: List[str], **kwargs) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, 0), dtype=float)
        return np.asarray(self.embed(texts, **kwargs), dtype=float).reshape(len(texts), -1)


class OllamaEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="qwen3", cache_dir=None, **kwargs):
        """
        Args:
            model_name (str): qwen3-embedding:latest, 
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model_kwargs = kwargs
        self.dimensions = self.model_kwargs.pop("dimensions", None)
    
    def embed(self, text):
        params = {
            "input": text,
            "options": self.model_kwargs,
            "model": self.model_name,
            "keep_alive": '10m',
        }
        if self.dimensions is not None:
            params["dimensions"] = self.dimensions
        embs = ollama.embed(**params).embeddings[0]
        return embs

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, 0), dtype=float)
        params = {
            "input": texts,
            "options": self.model_kwargs,
            "model": self.model_name,
            "keep_alive": '10m',
        }
        if self.dimensions is not None:
            params["dimensions"] = self.dimensions
        return np.asarray(ollama.embed(**params).embeddings, dtype=float)


class VLLMEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="Qwen/Qwen3-Embedding-8B", cache_dir=None, **kwargs):
        self.model_name = model_name
        self.model = None
        self.cache_dir = cache_dir
        self.model_kwargs = kwargs
        self.dimensions = self.model_kwargs.pop("dimensions", None)

    def load_model(self):
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:
                return
            try:
                from vllm import LLM
            except ImportError as e:
                raise ImportError("vllm is not installed.") from e

            init_kwargs = self.model_kwargs.copy()
            if self.cache_dir is not None:
                init_kwargs["download_dir"] = self.cache_dir
            self.model = LLM(model=self.model_name, task="embed", **init_kwargs)

    def embed(self, text):
        self.load_model()

        embed_kwargs = {}
        if self.dimensions is not None:
            embed_kwargs["dimensions"] = self.dimensions

        embs = [output.outputs.embedding for output in self.model.embed(text, **embed_kwargs)]
        if isinstance(text, str):
            return embs[0]
        return np.asarray(embs)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, 0), dtype=float)
        return np.asarray(self.embed(texts), dtype=float).reshape(len(texts), -1)


class TransformersEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="facebook/contriever", cache_dir=None, **kwargs):
        """
        Args:
            model_name (str): facebook/contriever, 
                              nvidia/NV-Embed-v2 (one GPU only), 
        """
        self.model_name = model_name
        self.model = None
        self.model_kwargs = kwargs
        self.tokenizer = None
        self.cache_dir = cache_dir
        if self.model_name in ("nvidia/NV-Embed-v2"):
            assert transformers.__version__ <= "4.46.0", "Use old transformers package (<= 4.46.0)."

    def load_model(self):
        if self.model is not None and self.tokenizer is not None:
            return
        with self._load_lock:
            self._load_model_unlocked()

    def _load_model_unlocked(self):
        if self.model is None:
            model_init_params = {
                "trust_remote_code": True,
                'device_map': "auto",  # added this line to use multiple GPUs
                "torch_dtype": "auto",
            }
            model_kwargs = self.model_kwargs.copy()
            model_kwargs.update(model_init_params)
            self.model = AutoModel.from_pretrained(self.model_name, 
                                                   mirror=os.environ["HF_ENDPOINT"], 
                                                   cache_dir=self.cache_dir,
                                                   **model_kwargs)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, 
                                                           mirror=os.environ["HF_ENDPOINT"],
                                                           cache_dir=self.cache_dir)

    def embed(self, text, **kwargs):
        self.load_model()
        if self.model_name in ("facebook/contriever"):
            return self._embed_contriever(text, **kwargs)
        elif self.model_name in ("nvidia/NV-Embed-v2"):
            return self._embed_nvidia(text, **kwargs)

    def embed_batch(self, texts: List[str], **kwargs) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, 0), dtype=float)
        if self.model_name in ("nvidia/NV-Embed-v2",):
            return np.asarray(self.embed(texts, **kwargs), dtype=float).reshape(len(texts), -1)
        return super().embed_batch(texts)

    def _embed_contriever(self, text, **kwargs):
        kwargs.setdefault("normalize", True)
        kwargs.setdefault("output_hidden_states", False)

        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**inputs)

        return outputs
    
    def _embed_nvidia(self, text: List[str], **kwargs):
        
        if isinstance(text, str):
            text = [text]
        kwargs.setdefault("instruction", "")
        kwargs.setdefault("batch_size", 4)
        kwargs.setdefault("num_workers", 32)
        kwargs.setdefault("norm", True)

        if len(text) <= kwargs["batch_size"]:
            kwargs["prompts"] = text 
            embs = self.model.encode(**kwargs)
        else:
            embs = []
            bar = tqdm(range(0, len(text), kwargs["batch_size"]), desc="creating leaf nodes")
            for i in bar:
                kwargs["prompts"] = text[i:i + kwargs["batch_size"]]
                embs.append(self.model.encode(**kwargs))
            bar.close()
            embs = torch.cat(embs, dim=0)
        
        if kwargs["norm"]:
            embs = normalize(embs, p=2, dim=1)

        return embs.squeeze().cpu().numpy()

