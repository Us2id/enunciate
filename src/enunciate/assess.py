"""End-to-end assessment: text and audio in, word-level feedback out."""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .align import align
from .audio import Audio, load_audio
from .config import Settings
from .g2p import Phonemizer, WordPhonemes
from .recognizer import HeardPhoneme, Recognizer
from .scoring import WordScore, score_words
from .variants import ENGLISH_RULES, STRICT_RULES, VariantRules, split_heard, split_rhotics


class AssessmentError(ValueError):
    """Input that cannot be assessed."""


class EmptyTextError(AssessmentError):
    pass


class TextTooLongError(AssessmentError):
    pass


class UnscorableTextError(AssessmentError):
    """None of the words produced phonemes."""


class AudioTooShortError(AssessmentError):
    pass


class AudioTooLongError(AssessmentError):
    pass


@dataclass(frozen=True)
class Assessment:
    text: str
    voice: str
    model: str
    audio_seconds: float
    expected: tuple[str, ...]
    heard: tuple[HeardPhoneme, ...]
    words: tuple[WordScore, ...]
    matches: int
    accepted: int
    substitutions: int
    deletions: int
    insertions: int
    phoneme_error_rate: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class Assessor:
    """Combines a :class:`Phonemizer` and a :class:`Recognizer` with input validation."""

    def __init__(
        self,
        recognizer: Recognizer,
        phonemizer: Phonemizer,
        *,
        rules: VariantRules = ENGLISH_RULES,
        max_audio_seconds: float = 30.0,
        min_audio_seconds: float = 0.2,
        max_text_chars: int = 500,
    ) -> None:
        self.recognizer = recognizer
        self.phonemizer = phonemizer
        self.rules = rules
        self.max_audio_seconds = max_audio_seconds
        self.min_audio_seconds = min_audio_seconds
        self.max_text_chars = max_text_chars

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> Assessor:
        settings = settings or Settings.from_env()
        return cls(
            Recognizer(settings.model_id, settings.device),
            Phonemizer(settings.voice, espeak_path=settings.espeak_path),
            rules=ENGLISH_RULES if settings.accept_variants else STRICT_RULES,
            max_audio_seconds=settings.max_audio_seconds,
            min_audio_seconds=settings.min_audio_seconds,
            max_text_chars=settings.max_text_chars,
        )

    def warm_up(self) -> None:
        """Load the model and prime it ahead of the first request."""
        self.recognizer.warm_up()

    def expected_phonemes(self, text: str) -> list[WordPhonemes]:
        """Validate ``text`` and return its words with expected phonemes."""
        if not text.strip():
            raise EmptyTextError("text is empty")
        if len(text) > self.max_text_chars:
            raise TextTooLongError(f"text is longer than {self.max_text_chars} characters")
        words = [
            WordPhonemes(w.word, tuple(split_rhotics(w.phonemes)))
            for w in self.phonemizer.phonemize(text)
            if w.phonemes
        ]
        if not words:
            raise UnscorableTextError("text contains no pronounceable words")
        return words

    def check_audio(self, audio: Audio) -> None:
        if audio.duration < self.min_audio_seconds:
            raise AudioTooShortError(
                f"audio is {audio.duration:.2f}s; at least {self.min_audio_seconds:g}s is needed"
            )
        if audio.duration > self.max_audio_seconds:
            raise AudioTooLongError(
                f"audio is {audio.duration:.1f}s; the limit is {self.max_audio_seconds:g}s"
            )

    def assess(self, audio: Audio, text: str) -> Assessment:
        words = self.expected_phonemes(text)
        self.check_audio(audio)
        heard = self.recognizer.recognize(audio)
        return self.assess_phonemes(words, heard, text=text, audio_seconds=audio.duration)

    def assess_phonemes(
        self,
        words: Sequence[WordPhonemes],
        heard: Sequence[HeardPhoneme],
        *,
        text: str,
        audio_seconds: float,
    ) -> Assessment:
        """Score already-recognised phonemes against expected words."""
        heard = split_heard(heard)
        expected = [p for w in words for p in w.phonemes]
        path = align(
            expected,
            [h.symbol for h in heard],
            substitution=self.rules.substitution_cost,
            insertion=self.rules.insertion_cost,
            deletion=self.rules.deletion_cost,
        )
        scored, totals = score_words(words, heard, path)
        return Assessment(
            text=text,
            voice=self.phonemizer.voice,
            model=self.recognizer.model_id,
            audio_seconds=round(audio_seconds, 3),
            expected=tuple(expected),
            heard=tuple(heard),
            words=tuple(scored),
            matches=totals.matches,
            accepted=totals.accepted,
            substitutions=totals.substitutions,
            deletions=totals.deletions,
            insertions=totals.insertions,
            phoneme_error_rate=round(totals.phoneme_error_rate, 4),
            score=round(totals.score, 4),
        )


_default_lock = threading.Lock()
_default: Assessor | None = None


def default_assessor(settings: Settings | None = None) -> Assessor:
    """Process-wide assessor built from ``settings`` (or the environment) on first use."""
    global _default
    with _default_lock:
        if _default is None:
            _default = Assessor.from_settings(settings)
        return _default


def assess(
    audio: str | Path | bytes | Audio, text: str, *, settings: Settings | None = None
) -> Assessment:
    """Convenience wrapper: ``assess("clip.wav", "I love India")``."""
    if not isinstance(audio, Audio):
        audio = load_audio(audio)
    return default_assessor(settings).assess(audio, text)
