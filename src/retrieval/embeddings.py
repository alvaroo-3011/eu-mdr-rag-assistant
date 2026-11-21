"""Validated local dense-embedding boundary for retrieval.

The adapter keeps model identity and inference behavior explicit. It performs
no chunk loading, index persistence, similarity search, or model generation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class EmbeddingError(ValueError):
    """Raised when embedding configuration or output is invalid."""


@dataclass(frozen=True, slots=True)
class EmbeddingModelConfig:
    """Complete identity and runtime contract for one embedding model."""

    model_id: str
    revision: str
    dimension: int
    max_sequence_length: int
    device: str
    normalize_embeddings: bool

    def __post_init__(self) -> None:
        """Validate explicit configuration without supplying hidden defaults."""

        for field_name in ("model_id", "revision", "device"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise EmbeddingError(
                    f"{field_name} must be a non-empty string."
                )

        for field_name in ("dimension", "max_sequence_length"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise EmbeddingError(
                    f"{field_name} must be a positive integer."
                )

        if not isinstance(self.normalize_embeddings, bool):
            raise EmbeddingError(
                "normalize_embeddings must be bool."
            )


# Pinned to the verified model revision inspected when the model decision was
# approved. Updating the revision must be treated as an index rebuild event.
GTE_MODERNBERT_BASE = EmbeddingModelConfig(
    model_id="Alibaba-NLP/gte-modernbert-base",
    revision="e7f32e3c00f91d699e8c43b53106206bcc72bb22",
    dimension=768,
    max_sequence_length=8_192,
    device="cpu",
    normalize_embeddings=True,
)


class _EmbeddingBackend(Protocol):
    """Minimal local embedding behavior used by this adapter."""

    max_seq_length: int

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> object:
        """Encode supplied text in order."""


BackendFactory = Callable[[EmbeddingModelConfig], _EmbeddingBackend]


class _TransformersBackend:
    """Direct Transformers implementation without scikit-learn imports."""

    def __init__(
        self,
        *,
        tokenizer: object,
        model: object,
        torch_module: object,
        functional_module: object,
        device: str,
        max_seq_length: int,
    ) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch_module
        self._functional = functional_module
        self._device = device
        self.max_seq_length = max_seq_length

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> object:
        """Encode ordered text batches with CLS pooling."""

        del show_progress_bar
        del convert_to_numpy

        batches: list[NDArray[np.float32]] = []

        for start in range(0, len(sentences), batch_size):
            batch = sentences[start:start + batch_size]
            encoded = self._tokenizer(
                batch,
                max_length=self.max_seq_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(self._device)
                for key, value in encoded.items()
            }

            with self._torch.inference_mode():
                outputs = self._model(**encoded)
                vectors = outputs.last_hidden_state[:, 0]
                if normalize_embeddings:
                    vectors = self._functional.normalize(
                        vectors,
                        p=2,
                        dim=1,
                    )

            batches.append(
                vectors.detach().cpu().numpy().astype(
                    np.float32,
                    copy=False,
                )
            )

        return np.concatenate(batches, axis=0)


def _load_transformers_backend(
    config: EmbeddingModelConfig,
) -> _EmbeddingBackend:
    """Load the pinned model directly through Transformers and PyTorch."""

    try:
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise EmbeddingError(
            "transformers and torch are required for local dense embeddings."
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            revision=config.revision,
            trust_remote_code=False,
            local_files_only=True,
        )
        model = AutoModel.from_pretrained(
            config.model_id,
            revision=config.revision,
            trust_remote_code=False,
            local_files_only=True,
        )
        model.to(config.device)
        model.eval()
    except Exception as exc:
        raise EmbeddingError(
            "Unable to load the pinned embedding model: "
            f"model_id={config.model_id!r}, revision={config.revision!r}."
        ) from exc

    return _TransformersBackend(
        tokenizer=tokenizer,
        model=model,
        torch_module=torch,
        functional_module=functional,
        device=config.device,
        max_seq_length=config.max_sequence_length,
    )


def _validated_texts(texts: Iterable[str]) -> tuple[str, ...]:
    """Materialize non-empty input text while preserving supplied order."""

    if isinstance(texts, (str, bytes)):
        raise EmbeddingError(
            "texts must be an iterable of strings, not one string."
        )

    try:
        values = tuple(texts)
    except TypeError as exc:
        raise EmbeddingError(
            "texts must be an iterable of strings."
        ) from exc

    if not values:
        raise EmbeddingError(
            "texts must contain at least one string."
        )

    for position, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise EmbeddingError(
                "texts contains a non-string or blank value at position "
                f"{position}."
            )

    return values


def _validated_batch_size(batch_size: int) -> int:
    """Require callers to choose an explicit positive batch size."""

    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise EmbeddingError(
            "batch_size must be a positive integer."
        )

    return batch_size


class TransformerEmbedder:
    """Lazy, CPU-first adapter around one pinned embedding model."""

    def __init__(
        self,
        config: EmbeddingModelConfig,
        *,
        backend_factory: BackendFactory | None = None,
    ) -> None:
        if not isinstance(config, EmbeddingModelConfig):
            raise EmbeddingError(
                "config must be an EmbeddingModelConfig."
            )
        if backend_factory is not None and not callable(backend_factory):
            raise EmbeddingError(
                "backend_factory must be callable or None."
            )

        self._config = config
        self._backend_factory = (
            backend_factory
            if backend_factory is not None
            else _load_transformers_backend
        )
        self._backend: _EmbeddingBackend | None = None

    @property
    def config(self) -> EmbeddingModelConfig:
        """Return the immutable model configuration."""

        return self._config

    def _get_backend(self) -> _EmbeddingBackend:
        """Create the model only on the first embedding request."""

        if self._backend is None:
            try:
                backend = self._backend_factory(self._config)
                backend.max_seq_length = self._config.max_sequence_length
            except EmbeddingError:
                raise
            except Exception as exc:
                raise EmbeddingError(
                    "Embedding backend initialization failed."
                ) from exc
            self._backend = backend

        return self._backend

    def embed_documents(
        self,
        texts: Iterable[str],
        *,
        batch_size: int,
    ) -> NDArray[np.float32]:
        """Embed document text in supplied order and validate the matrix."""

        values = _validated_texts(texts)
        validated_batch_size = _validated_batch_size(batch_size)
        backend = self._get_backend()

        try:
            raw_vectors = backend.encode(
                list(values),
                batch_size=validated_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=(
                    self._config.normalize_embeddings
                ),
            )
        except Exception as exc:
            raise EmbeddingError(
                "Embedding backend failed while encoding documents."
            ) from exc

        vectors = np.asarray(raw_vectors, dtype=np.float32)
        expected_shape = (len(values), self._config.dimension)

        if vectors.shape != expected_shape:
            raise EmbeddingError(
                "Embedding matrix has an unexpected shape: "
                f"expected={expected_shape}, actual={vectors.shape}."
            )
        if not np.isfinite(vectors).all():
            raise EmbeddingError(
                "Embedding matrix contains a non-finite value."
            )

        vectors.setflags(write=False)
        return vectors

    def embed_query(
        self,
        query: str,
    ) -> NDArray[np.float32]:
        """Embed one non-empty query and return one validated vector."""

        matrix = self.embed_documents(
            (query,),
            batch_size=1,
        )
        vector = matrix[0]
        vector.setflags(write=False)
        return vector


__all__ = [
    "EmbeddingError",
    "EmbeddingModelConfig",
    "GTE_MODERNBERT_BASE",
    "TransformerEmbedder",
]
