import pytest

from enunciate.align import AlignedPair, align
from enunciate.g2p import WordPhonemes
from enunciate.recognizer import HeardPhoneme
from enunciate.scoring import score_words

WORDS = [WordPhonemes("I", ("aɪ",)), WordPhonemes("love", ("l", "ʌ", "v"))]


def heard(symbols: list[str]) -> list[HeardPhoneme]:
    return [HeardPhoneme(s, i * 0.1, (i + 1) * 0.1) for i, s in enumerate(symbols)]


def test_perfect_match():
    h = heard(["aɪ", "l", "ʌ", "v"])
    scored, totals = score_words(WORDS, h, align(["aɪ", "l", "ʌ", "v"], [p.symbol for p in h]))
    assert [w.score for w in scored] == [1.0, 1.0]
    assert totals.phoneme_error_rate == 0.0
    assert totals.score == 1.0
    assert scored[1].heard == ("l", "ʌ", "v")
    assert scored[1].phonemes[0].start == pytest.approx(0.1)


def test_substitution_lowers_only_its_word():
    h = heard(["aɪ", "l", "ʌ", "b"])
    scored, totals = score_words(WORDS, h, align(["aɪ", "l", "ʌ", "v"], [p.symbol for p in h]))
    assert scored[0].score == 1.0
    assert scored[1].score == pytest.approx(2 / 3, abs=1e-4)
    assert scored[1].substitutions == 1
    assert scored[1].phonemes[2].expected == "v"
    assert scored[1].phonemes[2].heard == "b"
    assert totals.phoneme_error_rate == pytest.approx(0.25)


def test_insertion_is_charged_to_the_preceding_word():
    h = heard(["aɪ", "x", "l", "ʌ", "v"])
    scored, _ = score_words(WORDS, h, align(["aɪ", "l", "ʌ", "v"], [p.symbol for p in h]))
    assert scored[0].insertions == 1
    assert scored[0].score == 0.0
    assert scored[1].score == 1.0


def test_insertion_before_any_phoneme_goes_to_the_first_word():
    h = heard(["x", "aɪ", "l", "ʌ", "v"])
    scored, _ = score_words(WORDS, h, align(["aɪ", "l", "ʌ", "v"], [p.symbol for p in h]))
    assert scored[0].insertions == 1
    assert scored[1].errors == 0


def test_score_is_floored_at_zero():
    h = heard(["aɪ", "x", "y", "z", "l", "ʌ", "v"])
    scored, totals = score_words(WORDS, h, align(["aɪ", "l", "ʌ", "v"], [p.symbol for p in h]))
    assert scored[0].score == 0.0
    assert 0.0 <= totals.score <= 1.0


def test_nothing_heard_means_everything_deleted():
    scored, totals = score_words(WORDS, [], align(["aɪ", "l", "ʌ", "v"], []))
    assert [w.score for w in scored] == [0.0, 0.0]
    assert totals.deletions == 4
    assert scored[1].heard == ()


def test_accepted_variants_are_reported_but_not_counted():
    path = [
        AlignedPair("match", 0, 0, 0),
        AlignedPair("match", 1, 1, 0),
        AlignedPair("substitute", 2, 2, 0),  # accepted by the variant rules
        AlignedPair("match", 3, 3, 0),
    ]
    h = heard(["aɪ", "l", "ɐ", "v"])
    scored, totals = score_words(WORDS, h, path)
    assert scored[1].score == 1.0
    assert scored[1].accepted == 1
    assert scored[1].substitutions == 0
    assert scored[1].phonemes[1].accepted is True
    assert scored[1].phonemes[1].heard == "ɐ"
    assert totals.accepted == 1
    assert totals.errors == 0


def test_requires_words():
    with pytest.raises(ValueError):
        score_words([], [], [])
