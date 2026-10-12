"""Offline encoder checks with a tiny tokenizer and synthetic ONNX output."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from slipagent.semantic import DEFAULT_MODEL, ONNXEmbedder


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model_dir = self.root / DEFAULT_MODEL
        (self.model_dir / "onnx").mkdir(parents=True)
        (self.model_dir / "onnx/model_quantized.onnx").write_bytes(b"test model")
        (self.model_dir / "config.json").write_text(json.dumps({
            "hidden_size": 3, "max_position_embeddings": 6, "pad_token_id": 0,
        }))
        tokenizer = Tokenizer(models.WordPiece({
            "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "hello": 4, "world": 5,
        }, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
        tokenizer.save(str(self.model_dir / "tokenizer.json"))
        self.feeds = []
        self.session = SimpleNamespace(
            get_inputs=lambda: [SimpleNamespace(name=name) for name in
                                ("input_ids", "attention_mask", "token_type_ids")],
            run=self.run_model,
        )
        self.loader = self.enterContext(patch("onnxruntime.InferenceSession", return_value=self.session))

    def run_model(self, outputs, feed):
        self.assertEqual(outputs, ["last_hidden_state"])
        self.assertEqual(feed["input_ids"].dtype, np.int64)
        self.feeds.append(feed)
        ids = feed["input_ids"].astype(np.float32)
        return [np.stack([ids, np.ones_like(ids), -ids], axis=-1)]

    def test_mean_pooling_ignores_padding_and_normalizes_each_vector(self):
        embedder = ONNXEmbedder(model_dir=self.model_dir, batch_size=2)
        texts = ["hello", "hello world", "world"]
        vectors = embedder.embed_batch(texts)
        self.assertEqual([len(feed["input_ids"]) for feed in self.feeds], [2, 1])
        for vector, token_ids in zip(vectors, ([2, 4, 3], [2, 4, 5, 3], [2, 5, 3])):
            mean = np.mean(token_ids)
            expected = np.array([mean, 1, -mean])
            expected /= np.linalg.norm(expected)
            np.testing.assert_allclose(vector, expected, atol=1e-6)
        self.assertEqual(embedder.dim(), 3)
        self.loader.assert_called_once()
        self.assertEqual(self.loader.call_args.kwargs["providers"], ["CPUExecutionProvider"])

    def test_input_at_limit_is_complete_and_single_matches_batch(self):
        embedder = ONNXEmbedder(model_dir=self.model_dir)
        vector = embedder.embed("hello " * 4)
        self.assertEqual(self.feeds[-1]["input_ids"].shape, (1, 6))
        np.testing.assert_allclose(vector, embedder.embed_batch(["hello " * 4])[0])

    def test_default_embeds_each_source_independently_of_batch_neighbors(self):
        embedder = ONNXEmbedder(model_dir=self.model_dir)
        alone = embedder.embed("hello")
        vectors = embedder.embed_batch(["world " * 4, "hello", "world"])
        self.assertEqual(embedder.batch_size, 1)
        np.testing.assert_array_equal(vectors[1], alone)
        self.assertTrue(all(feed["input_ids"].shape[0] == 1 for feed in self.feeds))

    def test_empty_batch_does_not_load_or_download(self):
        embedder = ONNXEmbedder(model_dir=self.root / "missing")
        self.assertEqual(embedder.embed_batch([]), [])
        self.loader.assert_not_called()

    def test_oversized_input_is_rejected_before_inference(self):
        embedder = ONNXEmbedder(model_dir=self.model_dir)
        with self.assertRaisesRegex(ValueError, "token"):
            embedder.embed("hello " * 30)
        self.assertEqual(self.feeds, [])

    def test_indexing_splits_source_to_fit_and_keeps_the_last_lines(self):
        from slipagent.semantic import Searcher, Store

        workspace = self.root / "workspace"
        workspace.mkdir()
        source = workspace / "source.py"
        source.write_text("hello world\n" * 7)
        embedder = ONNXEmbedder(model_dir=self.model_dir)
        embedder.dim()
        # The filename and reserve need space in addition to the source tokens.
        embedder._max_tokens = 20
        with Store(str(self.root / "index.db"), dim=3) as store:
            searcher = Searcher(store, embedder)
            searcher.index_path(str(workspace))
            rows = list(store.conn.execute("SELECT start_line, end_line, text FROM chunks ORDER BY id"))
            self.assertGreater(len(rows), 1)
            self.assertEqual({line for row in rows for line in range(row["start_line"], row["end_line"] + 1)},
                             set(range(1, 8)))
            self.assertTrue(all(embedder.token_count("File: source.py\n\n" + row["text"]) <= 20 for row in rows))
            self.assertTrue(all(feed["input_ids"].shape[1] <= 20 for feed in self.feeds))

    def test_existing_conventions_cache_needs_no_hub_access(self):
        with patch.dict(os.environ, {"MEMORY_MODEL_CACHE_PATH": str(self.root)}), \
                patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("network access")):
            self.assertEqual(ONNXEmbedder().dim(), 3)
        self.assertEqual(Path(self.loader.call_args.args[0]), self.model_dir / "onnx/model_quantized.onnx")

    def test_explicit_missing_directory_does_not_download(self):
        with patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("network access")):
            with self.assertRaisesRegex(FileNotFoundError, "config.json"):
                ONNXEmbedder(model_dir=self.root / "missing").dim()

    def test_hub_cache_miss_downloads_only_missing_files_with_current_api(self):
        from huggingface_hub.errors import LocalEntryNotFoundError

        calls = []

        def download(repository, filename, **kwargs):
            calls.append((repository, filename, kwargs))
            if filename == "tokenizer.json" and kwargs.get("local_files_only"):
                raise LocalEntryNotFoundError("not cached")
            return str(self.model_dir / filename)

        with patch.dict(os.environ, {"MEMORY_MODEL_CACHE_PATH": str(self.root / "missing")}), \
                patch("huggingface_hub.hf_hub_download", side_effect=download):
            self.assertEqual(ONNXEmbedder().dim(), 3)
        self.assertEqual([filename for _, filename, options in calls if not options.get("local_files_only")],
                         ["tokenizer.json"])
        self.assertTrue(all(repository == DEFAULT_MODEL for repository, _, _ in calls))

    def test_invalid_batch_size_is_rejected(self):
        with self.assertRaises(ValueError):
            ONNXEmbedder(batch_size=0)

    def test_identity_changes_when_weights_or_tokenizer_change(self):
        identity = ONNXEmbedder(model_dir=self.model_dir).identity
        (self.model_dir / "onnx/model_quantized.onnx").write_bytes(b"different weights")
        new_identity = ONNXEmbedder(model_dir=self.model_dir).identity
        self.assertNotEqual(identity, new_identity)
        tokenizer_path = self.model_dir / "tokenizer.json"
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.add_tokens(["extra"])
        tokenizer.save(str(tokenizer_path))
        self.assertNotEqual(new_identity, ONNXEmbedder(model_dir=self.model_dir).identity)


if __name__ == "__main__":
    unittest.main()
