"""Phoneme-level English pronunciation assessment."""

from importlib.metadata import PackageNotFoundError, version

from .align import AlignedPair, align
from .assess import (
    Assessment,
    AssessmentError,
    Assessor,
    AudioTooLongError,
    AudioTooShortError,
    EmptyTextError,
    TextTooLongError,
    UnscorableTextError,
    assess,
    default_assessor,
)
from .audio import Audio, AudioDecodeError, load_audio
from .config import Settings
from .g2p import EspeakError, EspeakNotFoundError, Phonemizer, WordPhonemes
from .recognizer import HeardPhoneme, Recognizer
from .scoring import PhonemeScore, WordScore
from .variants import ENGLISH_RULES, STRICT_RULES, VariantRules

try:
    __version__ = version("enunciate")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0"

__all__ = [
    "ENGLISH_RULES",
    "STRICT_RULES",
    "AlignedPair",
    "Assessment",
    "AssessmentError",
    "Assessor",
    "Audio",
    "AudioDecodeError",
    "AudioTooLongError",
    "AudioTooShortError",
    "EmptyTextError",
    "EspeakError",
    "EspeakNotFoundError",
    "HeardPhoneme",
    "PhonemeScore",
    "Phonemizer",
    "Recognizer",
    "Settings",
    "TextTooLongError",
    "UnscorableTextError",
    "VariantRules",
    "WordPhonemes",
    "WordScore",
    "__version__",
    "align",
    "assess",
    "default_assessor",
    "load_audio",
]
