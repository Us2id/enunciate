"""Turn an alignment path into per-word and overall scores."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .align import AlignedPair, Op
from .g2p import WordPhonemes
from .recognizer import HeardPhoneme


@dataclass(frozen=True)
class PhonemeScore:
    """One aligned phoneme: what was expected, what was heard, and when.

    ``accepted`` is true for exact matches and for differences the variant
    rules treat as equivalent; only non-accepted steps count as errors.
    """

    op: Op
    expected: str | None
    heard: str | None
    accepted: bool
    start: float | None = None
    end: float | None = None


@dataclass(frozen=True)
class WordScore:
    word: str
    expected: tuple[str, ...]
    phonemes: tuple[PhonemeScore, ...]
    matches: int
    accepted: int
    substitutions: int
    deletions: int
    insertions: int
    score: float

    @property
    def heard(self) -> tuple[str, ...]:
        return tuple(p.heard for p in self.phonemes if p.heard is not None)

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


@dataclass(frozen=True)
class Totals:
    expected: int
    matches: int
    accepted: int
    substitutions: int
    deletions: int
    insertions: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def phoneme_error_rate(self) -> float:
        return self.errors / self.expected if self.expected else 0.0

    @property
    def score(self) -> float:
        return _clip(1.0 - self.phoneme_error_rate)


def score_words(
    words: Sequence[WordPhonemes],
    heard: Sequence[HeardPhoneme],
    path: Sequence[AlignedPair],
) -> tuple[list[WordScore], Totals]:
    """Project an alignment path onto words.

    Every expected phoneme belongs to exactly one word. Heard phonemes that
    match nothing (insertions) are charged to the word whose phoneme precedes
    them, or to the first word when they come before any expected phoneme.
    A word's score is ``1 - errors / expected phonemes``, floored at zero,
    where errors are the steps with a non-zero penalty.
    """
    if not words:
        raise ValueError("at least one word with expected phonemes is required")
    expected = [p for w in words for p in w.phonemes]
    owner = [index for index, w in enumerate(words) for _ in w.phonemes]
    buckets: list[list[PhonemeScore]] = [[] for _ in words]
    current = 0
    for pair in path:
        if pair.expected_index is not None:
            current = owner[pair.expected_index]
        heard_phoneme = heard[pair.heard_index] if pair.heard_index is not None else None
        buckets[current].append(
            PhonemeScore(
                op=pair.op,
                expected=expected[pair.expected_index] if pair.expected_index is not None else None,
                heard=heard_phoneme.symbol if heard_phoneme else None,
                accepted=pair.penalty == 0,
                start=heard_phoneme.start if heard_phoneme else None,
                end=heard_phoneme.end if heard_phoneme else None,
            )
        )

    scored = [_score_word(word, tuple(bucket)) for word, bucket in zip(words, buckets, strict=True)]
    totals = Totals(
        expected=len(expected),
        matches=sum(w.matches for w in scored),
        accepted=sum(w.accepted for w in scored),
        substitutions=sum(w.substitutions for w in scored),
        deletions=sum(w.deletions for w in scored),
        insertions=sum(w.insertions for w in scored),
    )
    return scored, totals


def _score_word(word: WordPhonemes, phonemes: tuple[PhonemeScore, ...]) -> WordScore:
    matches = sum(1 for p in phonemes if p.op == "match")
    accepted = sum(1 for p in phonemes if p.accepted and p.op != "match")
    errors = {
        op: sum(1 for p in phonemes if p.op == op and not p.accepted)
        for op in ("substitute", "delete", "insert")
    }
    total_errors = sum(errors.values())
    return WordScore(
        word=word.word,
        expected=word.phonemes,
        phonemes=phonemes,
        matches=matches,
        accepted=accepted,
        substitutions=errors["substitute"],
        deletions=errors["delete"],
        insertions=errors["insert"],
        score=round(_clip(1.0 - total_errors / len(word.phonemes)), 4) if word.phonemes else 0.0,
    )


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))
