import json
from pathlib import Path

from enunciate.recognizer import Recognizer, _load_vocabulary, resolve_device


def bare_recognizer() -> Recognizer:
    recognizer = Recognizer(model_id="fake/model", device="cpu")
    recognizer._tokens = ["<pad>", "<s>", "</s>", "<unk>", "a", "b", "c"]
    recognizer._blank_id = 0
    recognizer._frame_stride = 320
    return recognizer


def test_collapse_merges_repeats_and_drops_blanks():
    heard = bare_recognizer()._collapse([0, 4, 4, 0, 0, 5, 5, 5, 4, 0])
    assert [p.symbol for p in heard] == ["a", "b", "a"]
    assert [(p.start, p.end) for p in heard] == [(0.02, 0.06), (0.1, 0.16), (0.16, 0.18)]


def test_collapse_separates_identical_tokens_with_blank():
    heard = bare_recognizer()._collapse([4, 4, 0, 4])
    assert [p.symbol for p in heard] == ["a", "a"]


def test_collapse_drops_special_tokens_and_handles_edges():
    recognizer = bare_recognizer()
    assert recognizer._collapse([]) == []
    assert recognizer._collapse([0, 0, 0]) == []
    assert [p.symbol for p in recognizer._collapse([1, 4, 3, 6, 2])] == ["a", "c"]


def test_resolve_device_passthrough():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:1") == "cuda:1"
    assert resolve_device("auto") in {"cpu", "mps", "cuda"}


def test_load_vocabulary_from_local_directory(tmp_path: Path):
    (tmp_path / "vocab.json").write_text(json.dumps({"<pad>": 0, "b": 2, "a": 1}), encoding="utf-8")
    assert _load_vocabulary(str(tmp_path)) == ["<pad>", "a", "b"]


def test_recognizer_starts_unloaded():
    recognizer = Recognizer(model_id="fake/model")
    assert recognizer.is_loaded is False
    assert recognizer.device is None
