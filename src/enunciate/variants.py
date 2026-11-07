"""Accepted variation between espeak-ng's reference pronunciation and real speech.

The rules below were derived by running the recogniser over 73 LibriSpeech
dev-clean utterances (``scripts/evaluate.py``) and inspecting the most frequent
expected-to-heard confusions. Each rule forgives a difference that is a regular
feature of spoken General American English or an artefact of how the two
tools write the same sound, rather than a mispronunciation. Differences covered
by a rule are still reported, marked as accepted, so nothing is hidden.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from .recognizer import HeardPhoneme

#: espeak-ng writes an r-coloured vowel as one token before a consonant ("more"
#: → ɔːɹ) and as vowel + ɹ before a vowel ("more of" → ɔː ɹ), while the model
#: usually emits the fused token. Splitting both sides into vowel + ɹ makes the
#: two representations comparable.
SPLIT_TOKENS: Mapping[str, tuple[str, ...]] = {
    "ɑːɹ": ("ɑː", "ɹ"),
    "ɔːɹ": ("ɔː", "ɹ"),
    "oːɹ": ("oː", "ɹ"),
    "ɛɹ": ("ɛ", "ɹ"),
    "ɪɹ": ("ɪ", "ɹ"),
    "ʊɹ": ("ʊ", "ɹ"),
    "aɪɚ": ("aɪ", "ɚ"),
}

#: Pronunciations that espeak-ng gets wrong for the word on its own: the article
#: "a" is reduced in connected speech, but espeak-ng gives the letter name eɪ
#: when the word ends a clause.
WORD_OVERRIDES: Mapping[str, tuple[str, ...]] = {"a": ("ɐ",)}


def split_rhotics(symbols: Iterable[str]) -> list[str]:
    """Rewrite fused r-coloured vowels as vowel + ɹ."""
    out: list[str] = []
    for symbol in symbols:
        out.extend(SPLIT_TOKENS.get(symbol, (symbol,)))
    return out


def split_heard(heard: Iterable[HeardPhoneme]) -> list[HeardPhoneme]:
    """Same as :func:`split_rhotics` for recognised phonemes, keeping their timing."""
    out: list[HeardPhoneme] = []
    for phoneme in heard:
        for symbol in SPLIT_TOKENS.get(phoneme.symbol, (phoneme.symbol,)):
            out.append(HeardPhoneme(symbol, phoneme.start, phoneme.end))
    return out


@dataclass(frozen=True)
class VariantRules:
    """Zero-penalty differences used by :func:`enunciate.align.align`."""

    #: Groups of symbols that count as the same sound.
    equivalence_classes: tuple[frozenset[str], ...] = ()
    #: After these symbols an extra or missing ɹ carries no information.
    rhotic_context: frozenset[str] = frozenset()
    _equivalents: dict[str, frozenset[str]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        # A symbol may sit in several classes (ɾ is a variant of both t and d);
        # equivalence is per class and deliberately not transitive.
        for group in self.equivalence_classes:
            for symbol in group:
                self._equivalents[symbol] = self._equivalents.get(symbol, frozenset()) | group

    def substitution_cost(self, expected: str, heard: str) -> int:
        if expected == heard:
            return 0
        return 0 if heard in self._equivalents.get(expected, frozenset()) else 1

    def insertion_cost(self, previous_expected: str | None, heard: str) -> int:
        return 0 if heard == "ɹ" and previous_expected in self.rhotic_context else 1

    def deletion_cost(self, previous_expected: str | None, expected: str) -> int:
        return 0 if expected == "ɹ" and previous_expected in self.rhotic_context else 1


STRICT_RULES = VariantRules()

ENGLISH_RULES = VariantRules(
    equivalence_classes=(
        # cot–caught merger: most American speakers use one low back vowel for
        # "thought", "law", "on", "dog" and "father"; British speakers use ɒ, and
        # the model writes the vowel of "four" as oː for some speakers.
        frozenset({"ɔ", "ɔː", "ɑ", "ɑː", "ɒ", "oː"}),
        # The model writes the "oy" diphthong as oɪ where espeak-ng writes ɔɪ.
        frozenset({"ɔɪ", "oɪ"}),
        # Flapping: t and d between vowels are a tap in American English
        # ("water", "that is"), and espeak-ng applies it inconsistently.
        frozenset({"t", "ɾ"}),
        frozenset({"d", "ɾ"}),
        # espeak-ng's ᵻ marks a reduced vowel that is ɪ or ə in speech
        # ("roses", "wanted"); ɐ and ə are two spellings of the reduced vowel.
        frozenset({"ᵻ", "ɪ"}),
        frozenset({"ᵻ", "ə", "ɐ"}),
        # The r-coloured central vowel: espeak-ng writes ɜː when stressed
        # ("bird", "were") and ɚ when not; stress is not scored here.
        frozenset({"ɜː", "ɜ", "ɚ"}),
        # The unstressed final vowel of "happy": espeak-ng writes a short i, the
        # model often hears iː. (ɪ is kept distinct: "ship" is not "sheep".)
        frozenset({"i", "iː"}),
        # Syllabic consonants ("bottle", "button") with or without a schwa.
        frozenset({"əl", "l̩", "l"}),
        frozenset({"n̩", "ən", "n"}),
    ),
    rhotic_context=frozenset({"ɚ", "ɜː", "ɹ"}),
)
