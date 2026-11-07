"""Shared fixtures: fakes that keep unit tests independent of espeak-ng and the model."""

from __future__ import annotations

import numpy as np
import pytest

from enunciate import Assessor, Audio, HeardPhoneme, Phonemizer, Recognizer
from enunciate.g2p import tokenize

# Sentence-level espeak-ng output captured from espeak-ng 1.52 (``--ipa --sep=_``).
CANNED_SENTENCES: dict[str, str] = {
    "I love India": "aɪ l_ˈʌ_v ˈɪ_n_d_i__ə\n",
    "I": "aɪ\n",
    "the apple and the cat": "ð_ɪ_ ˈæ_p_əl__ æ_n_d ð_ə k_ˈæ_t\n",
    "of the middle": "ʌ_v_ð_ə m_ˈɪ_d_əl\n",
    "of": "ʌ_v\n",
    "the middle": "ð_ə m_ˈɪ_d_əl\n",
    "Hello, world. How are you?": "h_ə_l_ˈoʊ\nw_ˈɜː_l_d\nh_ˈaʊ ɑːɹ j_uː\n",
    "I have 2024 cats": "aɪ h_æ_v t_ˈuː θ_ˈaʊ_z_ə_n_d t_w_ˈɛ_n_t_i f_ˈɔː_ɹ k_ˈæ_t_s\n",
    "I have": "aɪ h_æ_v\n",
    "2024 cats": "t_ˈuː θ_ˈaʊ_z_ə_n_d t_w_ˈɛ_n_t_i f_ˈɔː_ɹ k_ˈæ_t_s\n",
    "2024": "t_ˈuː θ_ˈaʊ_z_ə_n_d t_w_ˈɛ_n_t_i f_ˈɔː_ɹ\n",
    "cats": "k_ˈæ_t_s\n",
    "a": "ˈeɪ\n",
    "car": "k_ˈɑːɹ\n",
    "nor": "n_ˈɔːɹ\n",
    "the car": "ð_ə k_ˈɑːɹ\n",
    "???": "\n",
}


class CannedRunner:
    """Stands in for the espeak-ng subprocess and records what it was asked."""

    def __init__(self, table: dict[str, str] | None = None) -> None:
        self.table = dict(CANNED_SENTENCES if table is None else table)
        self.calls: list[str] = []

    def __call__(self, text: str) -> str:
        self.calls.append(text)
        try:
            return self.table[text]
        except KeyError as exc:  # pragma: no cover - a test asked for an unknown sentence
            raise AssertionError(f"no canned espeak output for {text!r}") from exc


class FakeRecognizer(Recognizer):
    """Returns a fixed phoneme sequence instead of running the model."""

    def __init__(self, symbols: list[str]) -> None:
        super().__init__(model_id="fake/model", device="cpu")
        self.symbols = symbols
        self.device = "cpu"
        self.loaded = False

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    def load(self) -> None:
        self.loaded = True

    def warm_up(self) -> None:
        self.loaded = True

    def recognize(self, audio: Audio) -> list[HeardPhoneme]:
        step = audio.duration / max(len(self.symbols), 1)
        return [
            HeardPhoneme(symbol, round(i * step, 3), round((i + 1) * step, 3))
            for i, symbol in enumerate(self.symbols)
        ]


@pytest.fixture
def runner() -> CannedRunner:
    return CannedRunner()


@pytest.fixture
def phonemizer(runner: CannedRunner) -> Phonemizer:
    return Phonemizer("en-us", runner=runner)


@pytest.fixture
def silence() -> Audio:
    return Audio(np.zeros(16_000, dtype=np.float32), 16_000)


def make_assessor(
    heard: list[str], phonemizer: Phonemizer | None = None, **kwargs: object
) -> Assessor:
    return Assessor(
        FakeRecognizer(heard),
        phonemizer or Phonemizer("en-us", runner=CannedRunner()),
        **kwargs,  # type: ignore[arg-type]
    )


__all__ = ["CANNED_SENTENCES", "CannedRunner", "FakeRecognizer", "make_assessor", "tokenize"]
