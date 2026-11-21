"""Synthetic tests for the local dense-embedding boundary."""

from __future__ import annotations

import sys
import unittest
from dataclasses import FrozenInstanceError
from types import ModuleType
from unittest.mock import Mock, patch

import numpy as np

from src.retrieval.embeddings import (
    EmbeddingError,
    EmbeddingModelConfig,
    GTE_MODERNBERT_BASE,
    TransformerEmbedder,
    _load_transformers_backend,
)


class _FakeBackend:
    """Small deterministic replacement for the local model backend."""

    def __init__(
        self,
        dimension: int = 768,
        *,
        invalid_value: bool = False,
    ) -> None:
        self.dimension = dimension
        self.invalid_value = invalid_value
        self.max_seq_length = 512
        self.calls: list[dict[str, object]] = []

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        self.calls.append(
            {
                "sentences": sentences,
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
                "convert_to_numpy": convert_to_numpy,
                "normalize_embeddings": normalize_embeddings,
            }
        )
        vectors = np.zeros(
            (len(sentences), self.dimension),
            dtype=np.float32,
        )
        for index in range(len(sentences)):
            vectors[index, index % self.dimension] = 1.0

        if self.invalid_value:
            vectors[0, 0] = np.nan

        return vectors


class EmbeddingModelConfigTests(unittest.TestCase):
    """Validate explicit model identity and configuration."""

    def test_approved_model_is_pinned_and_long_context(self) -> None:
        self.assertEqual(
            GTE_MODERNBERT_BASE.model_id,
            "Alibaba-NLP/gte-modernbert-base",
        )
        self.assertEqual(
            GTE_MODERNBERT_BASE.revision,
            "e7f32e3c00f91d699e8c43b53106206bcc72bb22",
        )
        self.assertEqual(GTE_MODERNBERT_BASE.dimension, 768)
        self.assertEqual(
            GTE_MODERNBERT_BASE.max_sequence_length,
            8_192,
        )
        self.assertEqual(GTE_MODERNBERT_BASE.device, "cpu")
        self.assertTrue(GTE_MODERNBERT_BASE.normalize_embeddings)

    def test_config_is_immutable(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            GTE_MODERNBERT_BASE.dimension = 1  # type: ignore[misc]

    def test_invalid_config_values_are_rejected(self) -> None:
        valid = {
            "model_id": "model",
            "revision": "revision",
            "dimension": 3,
            "max_sequence_length": 10,
            "device": "cpu",
            "normalize_embeddings": True,
        }

        invalid_values = (
            {**valid, "model_id": " "},
            {**valid, "revision": ""},
            {**valid, "dimension": 0},
            {**valid, "dimension": True},
            {**valid, "max_sequence_length": -1},
            {**valid, "device": ""},
            {**valid, "normalize_embeddings": "yes"},
        )

        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises(EmbeddingError):
                    EmbeddingModelConfig(**values)


class TransformersBackendLoaderTests(unittest.TestCase):
    """Validate offline factory arguments without production dependencies."""

    def setUp(self) -> None:
        self.tokenizer = object()
        self.model = Mock()
        self.tokenizer_factory = Mock(return_value=self.tokenizer)
        self.model_factory = Mock(return_value=self.model)

        torch_module = ModuleType("torch")
        torch_module.__path__ = []  # type: ignore[attr-defined]
        nn_module = ModuleType("torch.nn")
        nn_module.__path__ = []  # type: ignore[attr-defined]
        functional_module = ModuleType("torch.nn.functional")
        torch_module.nn = nn_module  # type: ignore[attr-defined]
        nn_module.functional = functional_module  # type: ignore[attr-defined]

        transformers_module = ModuleType("transformers")
        transformers_module.AutoTokenizer = Mock(  # type: ignore[attr-defined]
            from_pretrained=self.tokenizer_factory,
        )
        transformers_module.AutoModel = Mock(  # type: ignore[attr-defined]
            from_pretrained=self.model_factory,
        )

        self.enterContext(patch.dict(sys.modules, {
            "torch": torch_module,
            "torch.nn": nn_module,
            "torch.nn.functional": functional_module,
            "transformers": transformers_module,
        }))

    def test_loads_pinned_model_from_local_files_only(self) -> None:
        backend = _load_transformers_backend(GTE_MODERNBERT_BASE)

        expected_call = {
            "revision": GTE_MODERNBERT_BASE.revision,
            "trust_remote_code": False,
            "local_files_only": True,
        }
        self.tokenizer_factory.assert_called_once_with(
            GTE_MODERNBERT_BASE.model_id,
            **expected_call,
        )
        self.model_factory.assert_called_once_with(
            GTE_MODERNBERT_BASE.model_id,
            **expected_call,
        )
        self.model.to.assert_called_once_with("cpu")
        self.model.eval.assert_called_once_with()
        self.assertEqual(backend.max_seq_length, 8_192)

    def test_missing_tokenizer_cache_retains_cause_and_stops_model_load(self) -> None:
        failure = OSError("synthetic tokenizer cache miss")
        self.tokenizer_factory.side_effect = failure

        with self.assertRaises(EmbeddingError) as caught:
            _load_transformers_backend(GTE_MODERNBERT_BASE)

        self.assertIs(caught.exception.__cause__, failure)
        self.model_factory.assert_not_called()

    def test_missing_model_cache_retains_cause(self) -> None:
        failure = OSError("synthetic model cache miss")
        self.model_factory.side_effect = failure

        with self.assertRaises(EmbeddingError) as caught:
            _load_transformers_backend(GTE_MODERNBERT_BASE)

        self.assertIs(caught.exception.__cause__, failure)


class TransformerEmbedderTests(unittest.TestCase):
    """Validate lazy loading, ordered encoding, and fail-closed output."""

    def test_backend_is_loaded_lazily_once(self) -> None:
        backend = _FakeBackend()
        factory_calls: list[EmbeddingModelConfig] = []

        def factory(config: EmbeddingModelConfig) -> _FakeBackend:
            factory_calls.append(config)
            return backend

        embedder = TransformerEmbedder(
            GTE_MODERNBERT_BASE,
            backend_factory=factory,
        )

        self.assertEqual(factory_calls, [])

        embedder.embed_query("first query")
        embedder.embed_query("second query")

        self.assertEqual(factory_calls, [GTE_MODERNBERT_BASE])
        self.assertEqual(backend.max_seq_length, 8_192)

    def test_documents_preserve_order_and_explicit_options(self) -> None:
        backend = _FakeBackend()
        embedder = TransformerEmbedder(
            GTE_MODERNBERT_BASE,
            backend_factory=lambda _: backend,
        )

        vectors = embedder.embed_documents(
            ("Article 1", "Annex I"),
            batch_size=2,
        )

        self.assertEqual(vectors.shape, (2, 768))
        self.assertEqual(vectors.dtype, np.float32)
        self.assertFalse(vectors.flags.writeable)
        self.assertEqual(
            backend.calls,
            [
                {
                    "sentences": ["Article 1", "Annex I"],
                    "batch_size": 2,
                    "show_progress_bar": False,
                    "convert_to_numpy": True,
                    "normalize_embeddings": True,
                }
            ],
        )

    def test_query_returns_one_read_only_vector(self) -> None:
        backend = _FakeBackend()
        embedder = TransformerEmbedder(
            GTE_MODERNBERT_BASE,
            backend_factory=lambda _: backend,
        )

        vector = embedder.embed_query("What does Article 15 require?")

        self.assertEqual(vector.shape, (768,))
        self.assertEqual(vector.dtype, np.float32)
        self.assertFalse(vector.flags.writeable)
        self.assertEqual(
            backend.calls[0]["sentences"],
            ["What does Article 15 require?"],
        )

    def test_invalid_text_inputs_are_rejected_before_loading(self) -> None:
        factory_calls = 0

        def factory(_: EmbeddingModelConfig) -> _FakeBackend:
            nonlocal factory_calls
            factory_calls += 1
            return _FakeBackend()

        embedder = TransformerEmbedder(
            GTE_MODERNBERT_BASE,
            backend_factory=factory,
        )

        invalid_inputs = (
            "one string",
            (),
            ("valid", " "),
            ("valid", 3),
        )

        for texts in invalid_inputs:
            with self.subTest(texts=texts):
                with self.assertRaises(EmbeddingError):
                    embedder.embed_documents(texts, batch_size=1)

        self.assertEqual(factory_calls, 0)

    def test_invalid_batch_sizes_are_rejected(self) -> None:
        embedder = TransformerEmbedder(
            GTE_MODERNBERT_BASE,
            backend_factory=lambda _: _FakeBackend(),
        )

        for batch_size in (0, -1, True, 1.5):
            with self.subTest(batch_size=batch_size):
                with self.assertRaises(EmbeddingError):
                    embedder.embed_documents(
                        ("evidence",),
                        batch_size=batch_size,  # type: ignore[arg-type]
                    )

    def test_wrong_output_dimension_is_rejected(self) -> None:
        embedder = TransformerEmbedder(
            GTE_MODERNBERT_BASE,
            backend_factory=lambda _: _FakeBackend(dimension=12),
        )

        with self.assertRaisesRegex(
            EmbeddingError,
            "unexpected shape",
        ):
            embedder.embed_query("query")

    def test_non_finite_output_is_rejected(self) -> None:
        embedder = TransformerEmbedder(
            GTE_MODERNBERT_BASE,
            backend_factory=lambda _: _FakeBackend(invalid_value=True),
        )

        with self.assertRaisesRegex(
            EmbeddingError,
            "non-finite",
        ):
            embedder.embed_query("query")

    def test_invalid_constructor_arguments_are_rejected(self) -> None:
        with self.assertRaises(EmbeddingError):
            TransformerEmbedder("config")  # type: ignore[arg-type]

        with self.assertRaises(EmbeddingError):
            TransformerEmbedder(
                GTE_MODERNBERT_BASE,
                backend_factory="factory",  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
