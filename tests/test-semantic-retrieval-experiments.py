"""Offline contracts for code coverage, lexical ranking, and held-out evaluation."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from slipagent.semantic.embed import ONNXEmbedder
from test_cli_e2e import cli_environment


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("retrieval_experiments", ROOT / "benchmarks/semantic/retrieval-experiments.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_empty_anchor_fails_instead_of_looping(tmp_path):
    for relative in ("codeindex/__init__.py", "tests/test-quality.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("sample source\n")
    catalog = {"query_styles": ["task"], "validation_files": [], "cases": [
        {"id": "empty", "path": "Dots/codeindex/__init__.py", "anchor": "", "queries": ["Find sample"]}
    ]}
    result = subprocess.run(
        [sys.executable, "-c", "import json, runpy, sys; from pathlib import Path; "
         "runpy.run_path(sys.argv[1])['freeze'](json.loads(sys.argv[2]), {'Dots': Path(sys.argv[3])})",
         str(ROOT / "benchmarks/semantic/freeze-realistic.py"), json.dumps(catalog), str(tmp_path)],
        env=cli_environment(), capture_output=True, text=True, timeout=10,
    )
    assert result.returncode != 0
    assert "ValueError: Empty source anchor: empty" in result.stderr


def fixture():
    source = "# heading\r\nclass Engine:\r\n    def refresh(self):\r\n        value = '" + "λ🌓" * 100 + "'\r\n        return value\r\n"
    anchor = "return value"
    start = source.index(anchor)
    target = {"path": "project/engine.py", "anchor": anchor, "start": start,
              "end": start + len(anchor), "line": source.count("\n", 0, start) + 1}
    return {"schema_version": 2,
            "files": [{"path": target["path"], "text": source, "sha256": hashlib.sha256(source.encode()).hexdigest()}],
            "queries": [{"id": "return/task", "family": "return", "style": "task", "split": "development",
                         "text": "Return the saved value", "targets": [target]}]}


class CorpusTests(unittest.TestCase):
    def test_all_policies_preserve_source_characters_unicode_line_endings_and_anchors(self):
        data = fixture()
        source = data["files"][0]["text"]
        for boundary in ("lines", "symbols"):
            for context in ("none", "path", "symbols"):
                with self.subTest(boundary=boundary, context=context):
                    corpus = experiment.build_corpus(data, lambda text: len(text) + 2,
                                                     budget=192, boundary=boundary, context=context)
                    coverage = set()
                    for doc in corpus["documents"]:
                        self.assertEqual(source[doc["start"]:doc["end"]], doc["text"])
                        self.assertLessEqual(len(doc["header"] + doc["text"]) + 2, 192)
                        coverage.update(range(doc["start"], doc["end"]))
                    self.assertEqual(coverage, set(range(len(source))))
                    relevant = corpus["queries"][0]["relevant"]
                    self.assertTrue(relevant)
                    self.assertTrue(all("return value" in corpus["documents"][index]["text"] for index in relevant))

    def test_symbol_context_comes_from_ast_including_decorators_and_nesting(self):
        source = "class A:\n    @decorate\n    async def work(self):\n        def nested():\n            pass\n        return nested()\n"
        ranges = experiment.symbols(source)
        self.assertEqual(experiment.source_header("a.py", ranges, 2, "symbols"),
                         "File: a.py\nSymbol: A.work\n\n")
        self.assertEqual(experiment.source_header("a.py", ranges, 5, "symbols"),
                         "File: a.py\nSymbol: A.work.nested\n\n")
        self.assertEqual(experiment.source_header("a.py", ranges, 5, "path"), "File: a.py\n\n")
        self.assertEqual(experiment.source_header("a.py", ranges, 5, "none"), "")

    def test_symbol_boundaries_keep_small_functions_separate(self):
        data = fixture()
        source = "def first():\n    return 1\n\ndef second():\n    return 2\n"
        data["files"][0]["text"] = source
        target = data["queries"][0]["targets"][0]
        target.update(anchor="return 2", start=source.index("return 2"), end=source.index("return 2") + 8, line=5)
        corpus = experiment.build_corpus(data, len, budget=512, boundary="symbols", context="none")
        answers = [corpus["documents"][index]["text"] for index in corpus["queries"][0]["relevant"]]
        self.assertEqual(answers, ["def second():\n    return 2\n"])

    def test_realistic_fixture_has_real_hashes_labels_styles_and_file_holdout(self):
        data, digest = experiment.load_dataset(ROOT / "benchmarks/semantic/realistic-search.json")
        self.assertEqual(digest, experiment.sha256(ROOT / "benchmarks/semantic/realistic-search.json"))
        self.assertGreaterEqual(len(data["queries"]), 250)
        self.assertGreaterEqual(len(data["files"]), 15)
        self.assertEqual({q["style"] for q in data["queries"]}, {"task", "question", "lookup"})
        families = {}
        for q in data["queries"]:
            families.setdefault(q["family"], []).append(q)
        self.assertTrue(all(len(items) == 3 for items in families.values()))

    def test_dataset_rejects_corrupt_source_and_cross_split_families_or_files(self):
        valid, _ = experiment.load_dataset(ROOT / "benchmarks/semantic/realistic-search.json")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            for mutation in ("source", "anchor", "family", "file"):
                data = copy.deepcopy(valid)
                if mutation == "source":
                    data["files"][0]["text"] += "corrupt"
                elif mutation == "anchor":
                    data["queries"][0]["targets"][0]["start"] += 1
                else:
                    data["queries"][1]["split"] = "validation"
                    if mutation == "file":
                        data["queries"][1]["family"] = "other-family"
                path.write_text(json.dumps(data))
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    experiment.load_dataset(path)


class KeywordTests(unittest.TestCase):
    def test_identifier_tokenization_matches_snake_camel_acronyms_and_unicode(self):
        self.assertEqual(experiment.words("HTTPServer read_command_output déjà"),
                         ["http", "server", "read", "command", "output", "déjà"])

    def test_bm25_favors_distinctive_matches_and_shorter_documents(self):
        index = experiment.KeywordIndex(["graph rollback", "graph unrelated " * 40, "clipboard copy"])
        scores, coverage, _ = index.score("rollback graph")
        self.assertGreater(scores[0], scores[1])
        self.assertEqual(scores[2], 0)
        self.assertGreater(coverage[0], coverage[1])
        np.testing.assert_array_equal(experiment.rank_candidates(np.zeros(3), scores, coverage, np.zeros(3), "bm25"), [0, 1, 2])

    def test_exact_identifiers_require_whole_lexemes_not_substrings(self):
        index = experiment.KeywordIndex(["def token_count(text): pass", "def token_counter(text): pass", "token count prose"])
        _, _, ids = index.score("Find token_count")
        np.testing.assert_array_equal(ids, [1, 0, 0])
        _, _, qualified = experiment.KeywordIndex(["Store.snapshot", "OtherStore.snapshot"]).score("Store.snapshot")
        np.testing.assert_array_equal(qualified, [1, 0])

    def test_quoted_phrase_has_exact_priority(self):
        index = experiment.KeywordIndex(["File changed while indexing; retry:", "File indexing changed for another reason"])
        lexical, coverage, identifiers = index.score('Find "File changed while indexing"')
        np.testing.assert_array_equal(identifiers, [1, 0])
        order = experiment.rank_candidates(np.array([0.1, 0.9]), lexical, coverage, identifiers, "identifiers-first")
        np.testing.assert_array_equal(order, [0, 1])

    def test_no_keyword_matches_preserve_semantic_order_in_fusion(self):
        semantic = np.array([0.2, 0.9, 0.5])
        zeros = np.zeros(3)
        for method in ("rrf", "hybrid-25", "hybrid-50", "hybrid-75", "words-first", "identifiers-first"):
            with self.subTest(method=method):
                np.testing.assert_array_equal(experiment.rank_candidates(semantic, zeros, zeros, zeros, method), [1, 2, 0])

    def test_equal_keyword_scores_receive_equal_fusion_credit(self):
        np.testing.assert_array_equal(experiment.reciprocal_ranks(np.zeros(3), positive_only=True), [0, 0, 0])
        ranks = experiment.reciprocal_ranks(np.array([2, 2, 1]))
        self.assertEqual(ranks[0], ranks[1])
        self.assertGreater(ranks[0], ranks[2])

    def test_word_match_priority_can_override_cosine(self):
        index = experiment.KeywordIndex(["rollback graph", "rollback", "something else"])
        lexical, coverage, identifiers = index.score("rollback graph")
        order = experiment.rank_candidates(np.array([0.1, 0.9, 0.8]), lexical, coverage, identifiers, "words-first")
        np.testing.assert_array_equal(order, [0, 1, 2])

    def test_unknown_ranker_is_rejected(self):
        with self.assertRaises(ValueError):
            experiment.rank_candidates(*[np.zeros(2)] * 4, "unknown")


class EvaluationTests(unittest.TestCase):
    def test_identical_inputs_are_embedded_once_in_deterministic_length_order(self):
        corpora = [{"documents": [{"header": "", "text": "long source"}, {"header": "", "text": "x"}]},
                   {"documents": [{"header": "", "text": "x"}, {"header": "File: f\n", "text": "x"}]}]
        queries = [{"text": "find source"}, {"text": "find source"}]
        actual = experiment.unique_inputs(corpora, queries, len)
        expected = {"long source", "x", "File: f\nx", "find source", experiment.quality.QUERY_PREFIX + "find source"}
        self.assertEqual(actual, sorted(expected, key=lambda text: (len(text), text)))

    def test_selection_never_reads_validation_metrics(self):
        conditions = [{"name": "a", "splits": {"development": {"mrr_at_10": 0.5, "hit_at_5": 1, "hit_at_1": 0.1}}},
                      {"name": "b", "splits": {"development": {"mrr_at_10": 0.4, "hit_at_5": 1, "hit_at_1": 0.9},
                                                "validation": {"mrr_at_10": 1}}}]
        self.assertEqual(experiment.choose(conditions)["name"], "a")
        conditions[0]["splits"]["validation"] = {"mrr_at_10": 0}
        self.assertEqual(experiment.choose(conditions)["name"], "a")

    def test_report_metrics_are_calculated_from_complete_rankings(self):
        data = fixture()
        corpus = experiment.build_corpus(data, len, budget=512, boundary="symbols", context="none")
        relevant = corpus["queries"][0]["relevant"][0]
        others = [i for i in range(len(corpus["documents"])) if i != relevant]
        report = experiment.measure(corpus, [np.array([others[0], relevant, *others[1:]])])
        self.assertEqual(report["metrics"]["hit_at_1"], 0)
        self.assertEqual(report["metrics"]["mrr_at_10"], 0.5)
        self.assertEqual(report["first_ranks"], [2])
        self.assertEqual(report["splits"]["development"]["hit_at_5"], 1)

    def test_vector_cache_reuses_exact_inputs_and_invalidates_changed_weights_or_text(self):
        calls = []
        encoder = SimpleNamespace(batch_size=8, both=lambda texts: calls.append(texts) or
                                  {name: np.array([[1., 0.]] * len(texts)) for name in ("mean", "cls")})
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            experiment.cached_vectors(encoder, ["a"], cache, {"weights": "first"})
            experiment.cached_vectors(encoder, ["a"], cache, {"weights": "first"})
            experiment.cached_vectors(encoder, ["b"], cache, {"weights": "first"})
            experiment.cached_vectors(encoder, ["a"], cache, {"weights": "second"})
            encoder.batch_size = 1
            experiment.cached_vectors(encoder, ["a"], cache, {"weights": "second"})
        self.assertEqual(calls, [["a"], ["b"], ["a"], ["a"]])


class EncoderTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "onnx").mkdir()
        (self.root / "onnx/model_quantized.onnx").write_bytes(b"q8")
        (self.root / "full.onnx").write_bytes(b"fp32")
        (self.root / "config.json").write_text(json.dumps({"hidden_size": 3, "max_position_embeddings": 6, "pad_token_id": 0}))
        tokenizer = Tokenizer(models.WordPiece({"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "hello": 4, "world": 5}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
        tokenizer.save(str(self.root / "tokenizer.json"))
        self.feeds = []

        def run(outputs, feed):
            self.feeds.append(feed)
            ids = feed["input_ids"].astype(np.float32)
            return [np.stack([ids, np.ones_like(ids), -ids], axis=-1)]

        session = SimpleNamespace(run=run, get_inputs=lambda: [SimpleNamespace(name=name) for name in
                                  ("input_ids", "attention_mask", "token_type_ids")])
        self.loader = self.enterContext(patch("onnxruntime.InferenceSession", return_value=session))

    def test_shared_forward_pass_matches_production_mean_and_normalized_cls(self):
        texts = ["hello", "hello world", "world " * 4]
        baseline = ONNXEmbedder(model_dir=self.root, batch_size=2)
        expected = baseline.embed_batch(texts)
        self.feeds.clear()
        encoder = experiment.ExperimentEncoder(self.root, self.root / "full.onnx", batch_size=2)
        result = encoder.both(texts)
        np.testing.assert_allclose(result["mean"], expected, atol=1e-7)
        np.testing.assert_allclose(result["cls"], [[2 / 3, 1 / 3, -2 / 3]] * 3, atol=1e-7)
        self.assertEqual(len(self.feeds), 2)
        self.assertEqual(Path(self.loader.call_args.args[0]), self.root / "full.onnx")

    def test_oversized_input_fails_before_inference(self):
        encoder = experiment.ExperimentEncoder(self.root, self.root / "full.onnx")
        with self.assertRaisesRegex(ValueError, "no truncation"):
            encoder.both(["world " * 20])
        self.assertEqual(self.feeds, [])

    def test_matrix_uses_the_two_threads_recorded_in_its_report(self):
        with patch("slipagent.semantic.embed.default_threads", return_value=7):
            encoder = experiment.ExperimentEncoder(self.root, self.root / "full.onnx")
        encoder.both(["hello"])
        self.assertEqual(self.loader.call_args.kwargs["sess_options"].intra_op_num_threads, 2)

    def test_explicit_missing_weights_fail_without_fallback(self):
        encoder = experiment.ExperimentEncoder(self.root, self.root / "absent.onnx")
        with self.assertRaises(FileNotFoundError):
            encoder.both(["hello"])
        self.loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
