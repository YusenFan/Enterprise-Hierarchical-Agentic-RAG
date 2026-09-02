"""
Single place that maps "<platform>:<model>" config names to backend classes.

Platforms: "ollama", "transformers", "sentence-transformers", "vllm", "api" (OpenAI-compatible),
plus the offline "hash" (embedding) and "fake" (abstract / qa) backends used by tests and smoke runs.
Task types: "embed", "abs", "qa", "rerank", "judge" ("judge" reuses the qa classes and reads
judge_model_kwargs / judge_cache_dir).
"""
from typing import Any, Dict

from .abstract import (OpenAIAbstractModel, OllamaAbstractModel, TransformersAbstractModel,
                       VLLMAbstractModel)
from .embed import (OpenAIEmbeddingModel, OllamaEmbeddingModel, SentenceTransformersEmbeddingModel,
                    TransformersEmbeddingModel, VLLMEmbeddingModel)
from .fake import FakeAbstractModel, FakeQAModel, HashEmbeddingModel
from .qa import OpenAIQAModel, OllamaQAModel, TransformersQAModel, VLLMQAModel
from .rerank import TransformersRerankModel, VLLMRerankModel

MODEL_CLASSES: Dict[str, Dict[str, type]] = {
    "ollama": {
        "embed": OllamaEmbeddingModel,
        "abs": OllamaAbstractModel,
        "qa": OllamaQAModel,
    },
    "transformers": {
        "embed": TransformersEmbeddingModel,
        "abs": TransformersAbstractModel,
        "qa": TransformersQAModel,
        "rerank": TransformersRerankModel,
    },
    "sentence-transformers": {
        "embed": SentenceTransformersEmbeddingModel,
    },
    "vllm": {
        "embed": VLLMEmbeddingModel,
        "abs": VLLMAbstractModel,
        "qa": VLLMQAModel,
        "rerank": VLLMRerankModel,
    },
    "api": {
        "embed": OpenAIEmbeddingModel,
        "abs": OpenAIAbstractModel,
        "qa": OpenAIQAModel,
    },
    "hash": {
        "embed": HashEmbeddingModel,
    },
    "fake": {
        "abs": FakeAbstractModel,
        "qa": FakeQAModel,
    },
}


def build_model(model_name: str, task_type: str, conf: Dict[str, Any]):
    """Instantiate the backend for `model_name` ("platform:model") and `task_type`."""
    if not isinstance(model_name, str) or ":" not in model_name:
        raise ValueError(
            f'Model name "{model_name}" must look like "<platform>:<model>", '
            f'e.g. "api:gpt-4o-mini" or "hash:bow".'
        )
    framework, name = model_name.split(":", maxsplit=1)
    class_task = "qa" if task_type == "judge" else task_type
    try:
        model_class = MODEL_CLASSES[framework][class_task]
    except KeyError:
        supported = sorted(
            f"{platform}:{task}" for platform, tasks in MODEL_CLASSES.items() for task in tasks
        )
        raise ValueError(
            f'No {task_type} backend for platform "{framework}". Supported platform:task pairs: {supported}'
        ) from None

    model_kwargs = dict(conf.get(f"{task_type}_model_kwargs", {}) or {})
    return model_class(name, cache_dir=conf.get(f"{task_type}_cache_dir", None), **model_kwargs)
