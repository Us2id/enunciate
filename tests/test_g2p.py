import pytest
from tests.conftest import CannedRunner

from enunciate.g2p import (
    EspeakNotFoundError,
    Phonemizer,
    WordPhonemes,
    _cut_point,
    display_form,
    tokenize,
)


def test_tokenize_keeps_only_tokens_with_letters_or_digits():
    assert tokenize("Hello, world - it's 9 o'clock ...") == [
        "Hello,",
        "world",
        "it's",
        "9",
        "o'clock",
    ]
    assert tokenize("   ") == []


def test_display_form_strips_edge_punctuation_only():
    assert display_form("world.") == "world"
    assert display_form("“Quilter's”") == "Quilter's"
    assert display_form("well-known,") == "well-known"
    assert display_form("...") == "..."


def test_parse_strips_stress_drops_empty_fields_and_joins_lines():
    raw = "aɪ l_ˈʌ_v ˈɪ_n_d_i__ə\n"
    assert Phonemizer.parse(raw) == [["aɪ"], ["l", "ʌ", "v"], ["ɪ", "n", "d", "i", "ə"]]
    assert Phonemizer.parse("ð_ɪ_ ˈæ_p_əl__\n") == [["ð", "ɪ"], ["æ", "p", "əl"]]
    assert Phonemizer.parse("h_ə_l_ˈoʊ\nw_ˈɜː_l_d\n") == [
        ["h", "ə", "l", "oʊ"],
        ["w", "ɜː", "l", "d"],
    ]
    assert Phonemizer.parse("") == []


def test_parse_removes_language_flags_and_joiners():
    assert Phonemizer.parse("(en) b_ˈɔː_n (fr) ʒ_ˈuː_ɹ\n") == [["b", "ɔː", "n"], ["ʒ", "uː", "ɹ"]]
    assert Phonemizer.parse("a‍ɪ t͡ʃ\n") == [["aɪ"], ["tʃ"]]


def test_phonemize_uses_sentence_context(runner: CannedRunner):
    words = Phonemizer("en-us", runner=runner).phonemize("the apple and the cat")
    assert [w.word for w in words] == ["the", "apple", "and", "the", "cat"]
    assert words[0].phonemes == ("ð", "ɪ")  # before a vowel
    assert words[3].phonemes == ("ð", "ə")  # before a consonant
    assert runner.calls == ["the apple and the cat"]


def test_phonemize_bisects_when_espeak_merges_words(runner: CannedRunner):
    words = Phonemizer("en-us", runner=runner).phonemize("of the middle")
    assert words == [
        WordPhonemes("of", ("ʌ", "v")),
        WordPhonemes("the", ("ð", "ə")),
        WordPhonemes("middle", ("m", "ɪ", "d", "əl")),
    ]
    assert runner.calls == ["of the middle", "of", "the middle"]


def test_phonemize_merges_expanded_tokens_into_one_word(runner: CannedRunner):
    words = Phonemizer("en-us", runner=runner).phonemize("I have 2024 cats")
    assert [w.word for w in words] == ["I", "have", "2024", "cats"]
    assert words[2].phonemes == (
        "t",
        "uː",
        "θ",
        "aʊ",
        "z",
        "ə",
        "n",
        "d",
        "t",
        "w",
        "ɛ",
        "n",
        "t",
        "i",
        "f",
        "ɔː",
        "ɹ",
    )
    assert words[3].phonemes == ("k", "æ", "t", "s")


def test_phonemize_applies_word_overrides(runner: CannedRunner):
    assert Phonemizer("en-us", runner=runner).phonemize("a") == [WordPhonemes("a", ("ɐ",))]
    assert Phonemizer("en-us", runner=runner, overrides=None).phonemize("a") == [
        WordPhonemes("a", ("eɪ",))
    ]


def test_phonemize_empty_and_unpronounceable(runner: CannedRunner):
    phonemizer = Phonemizer("en-us", runner=runner)
    assert phonemizer.phonemize("   ") == []
    assert phonemizer.phonemize("- ... !") == []


def test_phonemize_caches_chunks(runner: CannedRunner):
    phonemizer = Phonemizer("en-us", runner=runner)
    phonemizer.phonemize("I love India")
    phonemizer.phonemize("I love India")
    assert runner.calls == ["I love India"]


def test_cut_point_prefers_clause_boundary_near_the_middle():
    assert _cut_point(["a", "b", "c", "d"]) == 2
    assert _cut_point(["a,", "b", "c", "d", "e."]) == 1
    assert _cut_point(["a", "b", "c.", "d", "e", "f", "g"]) == 3


def test_missing_binary_raises_a_helpful_error():
    with pytest.raises(EspeakNotFoundError, match="brew install espeak-ng"):
        Phonemizer("en-us", espeak_path="/definitely/not/espeak-ng").phonemize("hello")


requires_espeak = pytest.mark.skipif(
    not Phonemizer.is_available(), reason="espeak-ng not installed"
)


@requires_espeak
def test_live_espeak_matches_canned_output():
    words = Phonemizer("en-us").phonemize("I love India")
    assert words == [
        WordPhonemes("I", ("aɪ",)),
        WordPhonemes("love", ("l", "ʌ", "v")),
        WordPhonemes("India", ("ɪ", "n", "d", "i", "ə")),
    ]


@requires_espeak
def test_live_espeak_keeps_last_word_intact():
    # espeak-ng 1.52 drops the final letter of an unterminated last word; the
    # runner always appends a newline.
    assert Phonemizer("en-us").phonemize("bath")[0].phonemes == ("b", "æ", "θ")
    assert Phonemizer("en-gb").phonemize("bath")[0].phonemes == ("b", "a", "θ")


@requires_espeak
def test_live_espeak_bisection_on_merged_function_words():
    words = Phonemizer("en-us").phonemize("the apostle of the middle classes")
    assert [w.word for w in words] == ["the", "apostle", "of", "the", "middle", "classes"]
    assert all(w.phonemes for w in words)
