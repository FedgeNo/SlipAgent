"""Offline checks for quality scoring, corpus labels, and experimental pooling."""

import copy
import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from slipagent.semantic import ONNXEmbedder
from slipagent.semantic.indexer import chunk_text


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("embedding_quality", ROOT / "benchmarks/semantic/embedding-quality.py")
quality = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quality)


def fixture():
    return {"schema_version": 1, "name": "test", "documents": [
        {"id": "a", "path": "a.py", "start_line": 1, "end_line": 1, "text": "alpha"},
        {"id": "b", "path": "b.py", "start_line": 1, "end_line": 1, "text": "beta"},
    ], "queries": [
        {"id": "q", "category": "test", "text": "find beta", "relevant": ["b"]},
    ]}


class QualityTests(unittest.TestCase):
    def test_metrics_match_hand_calculated_multiple_relevant_results(self):
        metrics = quality.ranking_metrics(["a", "b", "c", "d"], {"b", "d"})
        self.assertEqual(metrics["hit_at_1"], 0)
        self.assertEqual(metrics["hit_at_5"], 1)
        self.assertEqual(metrics["recall_at_5"], 1)
        self.assertEqual(metrics["mrr_at_10"], 0.5)
        expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
        self.assertAlmostEqual(metrics["ndcg_at_10"], expected)

    def test_recall_counts_all_relevant_chunks(self):
        metrics = quality.ranking_metrics(list("abcdef"), {"a", "f"})
        self.assertEqual(metrics["hit_at_1"], 1)
        self.assertEqual(metrics["recall_at_5"], 0.5)

    def test_rank_beyond_ten_is_a_miss_for_truncated_metrics(self):
        metrics = quality.ranking_metrics([str(index) for index in range(11)], {"10"})
        self.assertEqual(set(metrics.values()), {0.0})

    def test_invalid_rankings_are_rejected(self):
        for ranking, relevant in ((["a"], set()), (["a"], {"b"}), (["a", "a"], {"a"})):
            with self.subTest(ranking=ranking, relevant=relevant), self.assertRaises(ValueError):
                quality.ranking_metrics(ranking, relevant)

    def test_exact_cosine_ranking_normalizes_vectors_and_keeps_negative_scores(self):
        data = fixture()
        result = quality.evaluate(data["documents"], data["queries"], [[-2, 0], [0, 4]], [[0, 3]])
        self.assertEqual(result["metrics"]["hit_at_1"], 1)
        top = result["queries"][0]["top_10"]
        self.assertEqual([row["id"] for row in top], ["b", "a"])
        self.assertAlmostEqual(top[0]["cosine_similarity"], 1)
        negative = quality.evaluate(data["documents"], data["queries"], [[1, 0], [0, 1]], [[0, -1]])
        self.assertEqual(negative["queries"][0]["relevant_ranks"], {"b": 2})
        self.assertEqual(negative["queries"][0]["top_10"][1]["cosine_similarity"], -1)

    def test_ties_keep_corpus_order_without_favoring_relevant_documents(self):
        data = fixture()
        result = quality.evaluate(data["documents"], data["queries"], [[1, 0], [1, 0]], [[1, 0]])
        self.assertEqual(result["queries"][0]["relevant_ranks"], {"b": 2})
        self.assertEqual(result["metrics"]["hit_at_1"], 0)

    def test_invalid_vectors_and_mismatched_dimensions_are_rejected(self):
        data = fixture()
        for vectors in ([[0, 0], [1, 0]], [[float("nan"), 0], [1, 0]], [[1, 0]]):
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                quality.evaluate(data["documents"], data["queries"], vectors, [[1, 0]])
        with self.assertRaisesRegex(ValueError, "dimensions"):
            quality.evaluate(data["documents"], data["queries"], [[1, 0], [0, 1]], [[1, 0, 0]])

    def test_prefix_applies_only_to_queries_and_document_embeddings_are_reused(self):
        data = fixture()
        calls = []

        def encode(texts):
            calls.append(texts)
            return [[1, 0] for _ in texts]

        encoder = SimpleNamespace(pooling="mean", embed_batch=encode)
        results = quality.run_comparison(data, encoder)
        self.assertEqual(calls, [["alpha", "beta"], ["find beta"], [quality.QUERY_PREFIX + "find beta"]])
        self.assertEqual([row["name"] for row in results], ["mean-plain", "mean-prefix"])

    def test_dataset_validation_and_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.json"
            raw = json.dumps(fixture()).encode()
            path.write_bytes(raw)
            data, digest = quality.load_dataset(path)
            self.assertEqual(data, fixture())
            self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
            invalid = []
            for labels in ([], ["missing"], ["b", "b"]):
                changed = copy.deepcopy(fixture())
                changed["queries"][0]["relevant"] = labels
                invalid.append(changed)
            changed = fixture()
            changed["documents"][1]["id"] = "a"
            invalid.append(changed)
            changed = fixture()
            changed["queries"].append(copy.deepcopy(changed["queries"][0]))
            invalid.append(changed)
            for changed in invalid:
                path.write_text(json.dumps(changed))
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    quality.load_dataset(path)

    def test_short_queries_preserve_sources_and_labels_and_meet_word_limit(self):
        original, _ = quality.load_dataset(ROOT / "benchmarks/semantic/code-search.json")
        path = ROOT / "benchmarks/semantic/short-queries.json"
        short, digest = quality.replace_queries(original, path)
        self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(short["documents"], original["documents"])
        self.assertEqual(len(short["queries"]), 39)
        for new, old in zip(short["queries"], original["queries"]):
            self.assertIn(len(new["text"].split()), (5, 6))
            self.assertEqual({key: value for key, value in new.items() if key != "text"},
                             {key: value for key, value in old.items() if key != "text"})
            self.assertNotEqual(new["text"], old["text"])

    def test_replacement_queries_reject_misaligned_ids_and_invalid_word_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.json"
            invalid = [[], [{"id": "missing", "text": "Find matching source code in index"}],
                       [{"id": "q", "text": "too short"}],
                       [{"id": "q", "text": "Find matching source code in index"}] * 2]
            for queries in invalid:
                path.write_text(json.dumps({"schema_version": 1, "queries": queries}))
                with self.subTest(queries=queries), self.assertRaises(ValueError):
                    quality.replace_queries(fixture(), path)

    def test_frozen_corpus_hashes_chunking_and_answer_anchors(self):
        data, _ = quality.load_dataset(ROOT / "benchmarks/semantic/code-search.json")
        self.assertEqual(len(data["queries"]), 39)
        reconstructed = {}
        for source in data["sources"]:
            documents = [doc for doc in data["documents"] if doc["path"] == source["path"]]
            lines = {}
            for document in documents:
                text_lines = document["text"].splitlines(keepends=True)
                self.assertEqual(len(text_lines), document["end_line"] - document["start_line"] + 1)
                for number, text in enumerate(text_lines, document["start_line"]):
                    if number in lines:
                        self.assertEqual(lines[number], text)
                    lines[number] = text
            self.assertEqual(sorted(lines), list(range(1, max(lines) + 1)))
            text = "".join(lines[number] for number in sorted(lines))
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), source["sha256"])
            expected = chunk_text(source["path"], text, **data["chunking"])
            self.assertEqual([chunk.text for chunk in expected], [doc["text"] for doc in documents])
            reconstructed[source["path"]] = text.splitlines()
        for query in data["queries"]:
            target = query["target"]
            self.assertIn(target["anchor"], reconstructed[target["path"]][target["line"] - 1])
            expected = {doc["id"] for doc in data["documents"]
                        if doc["path"] == target["path"] and doc["start_line"] <= target["line"] <= doc["end_line"]}
            self.assertEqual(set(query["relevant"]), expected)

    def test_missing_local_model_fails_without_download_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("network access")), \
                    redirect_stderr(StringIO()) as stderr, self.assertRaises(SystemExit) as error:
                quality.main(["--model-dir", str(Path(directory) / "missing"), "--output", str(report)])
            self.assertEqual(error.exception.code, 2)
            self.assertIn("Missing embedding model file", stderr.getvalue())
            self.assertFalse(report.exists())


class PoolingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.model_dir = Path(directory.name)
        (self.model_dir / "onnx").mkdir()
        (self.model_dir / "onnx/model_quantized.onnx").write_bytes(b"fake")
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

        def run(outputs, feed):
            self.assertEqual(outputs, ["last_hidden_state"])
            self.feeds.append(feed)
            ids = feed["input_ids"].astype(np.float32)
            return [np.stack([ids, np.ones_like(ids), -ids], axis=-1)]

        session = SimpleNamespace(run=run, get_inputs=lambda: [SimpleNamespace(name=name) for name in
                                  ("input_ids", "attention_mask", "token_type_ids")])
        self.loader = self.enterContext(patch("onnxruntime.InferenceSession", return_value=session))

    def test_mean_baseline_matches_unmodified_production_encoder(self):
        texts = ["hello", "hello world", "world " * 4]
        baseline = ONNXEmbedder(model_dir=self.model_dir, batch_size=2)
        experiment = quality.BenchmarkEmbedder(model_dir=self.model_dir, pooling="mean", batch_size=2)
        np.testing.assert_array_equal(experiment.embed_batch(texts), baseline.embed_batch(texts))

    def test_cls_uses_first_token_normalization_and_batching(self):
        encoder = quality.BenchmarkEmbedder(model_dir=self.model_dir, pooling="cls", batch_size=2)
        vectors = encoder.embed_batch(["hello", "hello world", "world " * 4])
        expected = np.array([2, 1, -2], dtype=np.float32) / 3
        np.testing.assert_allclose(vectors, [expected] * 3)
        self.assertEqual([feed["input_ids"].shape for feed in self.feeds], [(2, 4), (1, 6)])
        self.assertTrue(all(feed["input_ids"].dtype == np.int64 for feed in self.feeds))

    def test_cls_rejects_oversized_input_before_inference(self):
        encoder = quality.BenchmarkEmbedder(model_dir=self.model_dir, pooling="cls")
        with self.assertRaisesRegex(ValueError, "token limit"):
            encoder.embed("world " * 30)
        self.assertEqual(self.feeds, [])

    def test_rechunked_corpus_covers_the_complete_source_and_relabels_answers(self):
        data = {**fixture(), "chunking": {"max_lines": 50, "overlap": 1}}
        source = "hello world\n" * 7
        data["documents"] = [{"id": "original", "path": "source.py", "start_line": 1,
                              "end_line": 7, "text": source}]
        data["queries"][0].update(relevant=["original"], target={"path": "source.py", "line": 7,
                                                               "anchor": "hello world"})
        encoder = quality.BenchmarkEmbedder(model_dir=self.model_dir, pooling="mean")
        result = quality.token_dataset(data, encoder)
        self.assertGreater(len(result["documents"]), 1)
        self.assertTrue(all(encoder.token_count(doc["text"]) <= 6 for doc in result["documents"]))
        lines = {}
        for doc in result["documents"]:
            for number, text in enumerate(doc["text"].splitlines(keepends=True), doc["start_line"]):
                lines[number] = text
        self.assertEqual("".join(lines[number] for number in sorted(lines)), source)
        self.assertTrue(result["queries"][0]["relevant"])
        self.assertEqual(data["queries"][0]["relevant"], ["original"])

    def test_empty_cls_batch_does_not_load(self):
        encoder = quality.BenchmarkEmbedder(model_dir=self.model_dir, pooling="cls")
        self.assertEqual(encoder.embed_batch([]), [])
        self.loader.assert_not_called()

    def test_truncation_diagnostics_do_not_change_encoder_tokenizer(self):
        encoder = quality.BenchmarkEmbedder(model_dir=self.model_dir, pooling="cls")
        encoder.dim()
        data = fixture()
        data["documents"][0]["text"] = "hello " * 30
        before = encoder._tokenizer.to_str()
        counts = quality.truncation_counts(data, encoder)
        self.assertEqual(counts["documents"], 1)
        self.assertEqual(counts["plain_queries"], 0)
        self.assertEqual(counts["prefixed_queries"], 1)
        self.assertEqual(encoder._tokenizer.to_str(), before)


if __name__ == "__main__":
    unittest.main()
