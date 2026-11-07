import json

import numpy as np
import pytest

from enunciate import (
    STRICT_RULES,
    Audio,
    AudioTooLongError,
    AudioTooShortError,
    EmptyTextError,
    TextTooLongError,
    UnscorableTextError,
)
from tests.conftest import make_assessor


def test_assess_returns_word_level_feedback(silence: Audio):
    assessment = make_assessor(["aɪ", "l", "ʌ", "b", "ɪ", "n", "d", "i", "ə"]).assess(
        silence, "I love India"
    )
    assert assessment.text == "I love India"
    assert [w.word for w in assessment.words] == ["I", "love", "India"]
    assert [w.score for w in assessment.words] == [1.0, pytest.approx(2 / 3, abs=1e-4), 1.0]
    assert assessment.substitutions == 1
    assert assessment.phoneme_error_rate == pytest.approx(1 / 9, abs=1e-4)
    assert assessment.score == pytest.approx(8 / 9, abs=1e-4)
    assert assessment.audio_seconds == 1.0
    assert assessment.model == "fake/model"
    assert assessment.voice == "en-us"


def test_assessment_is_json_serialisable(silence: Audio):
    assessment = make_assessor(["aɪ", "l", "ʌ", "v"]).assess(silence, "I love India")
    data = json.loads(assessment.to_json())
    assert data["words"][1]["phonemes"][0] == {
        "op": "match",
        "expected": "l",
        "heard": "l",
        "accepted": True,
        "start": 0.25,
        "end": 0.5,
    }
    assert data["words"][2]["deletions"] == 5
    assert assessment.to_dict()["heard"][0]["symbol"] == "aɪ"


def test_variants_are_accepted_by_default_and_counted_in_strict_mode(silence: Audio):
    heard = ["ð", "ə", "k", "ɑː", "ɹ"]  # espeak-ng says ɑːɹ as one token; the rules split it
    lenient = make_assessor(heard).assess(silence, "the car")
    assert lenient.expected == ("ð", "ə", "k", "ɑː", "ɹ")
    assert lenient.score == 1.0

    heard = ["ð", "ɐ", "k", "ɑː", "ɹ"]  # ɐ for ə is an accepted reduced-vowel spelling
    lenient = make_assessor(heard).assess(silence, "the car")
    assert lenient.score == 1.0
    assert lenient.accepted == 1
    assert lenient.words[0].phonemes[1].accepted is True
    strict = make_assessor(heard, rules=STRICT_RULES).assess(silence, "the car")
    assert strict.accepted == 0
    assert strict.substitutions == 1


def test_fused_rhotic_in_heard_output_is_split(silence: Audio):
    assessment = make_assessor(["ð", "ə", "k", "ɑːɹ"]).assess(silence, "the car")
    assert [h.symbol for h in assessment.heard] == ["ð", "ə", "k", "ɑː", "ɹ"]
    assert assessment.score == 1.0


def test_text_validation(silence: Audio):
    assessor = make_assessor(["aɪ"], max_text_chars=20)
    with pytest.raises(EmptyTextError):
        assessor.assess(silence, "   ")
    with pytest.raises(TextTooLongError):
        assessor.assess(silence, "x" * 21)
    with pytest.raises(UnscorableTextError):
        assessor.assess(silence, "???")


def test_audio_validation():
    assessor = make_assessor(["aɪ"], min_audio_seconds=0.5, max_audio_seconds=2.0)
    with pytest.raises(AudioTooShortError):
        assessor.assess(Audio(np.zeros(1600, dtype=np.float32), 16_000), "I")
    with pytest.raises(AudioTooLongError):
        assessor.assess(Audio(np.zeros(16_000 * 3, dtype=np.float32), 16_000), "I")


def test_warm_up_loads_the_recognizer(silence: Audio):
    assessor = make_assessor(["aɪ"])
    assert assessor.recognizer.is_loaded is False
    assessor.warm_up()
    assert assessor.recognizer.is_loaded is True
