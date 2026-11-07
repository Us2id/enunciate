"""Reference pronunciations from espeak-ng.

The recogniser was trained on transcripts phonemised by espeak-ng, so using
the same program for the expected pronunciation keeps both sides in one symbol
set. espeak-ng is invoked as a separate process, which keeps this package free
of GPL-licensed Python dependencies.
"""

from __future__ import annotations

import re
import shutil
import string
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass

DEFAULT_VOICE = "en-us"
SUPPORTED_VOICES: tuple[str, ...] = ("en-us", "en-gb")

_SEPARATOR = "_"
# Primary and secondary stress, zero-width joiner and tie bars carry no segmental information.
_DROP_CHARS = str.maketrans("", "", "ˈˌ‍͜͡")
_LANGUAGE_FLAG = re.compile(r"^\([a-z0-9_-]+\)$", re.IGNORECASE)
_HAS_ALNUM = re.compile(r"[^\W_]")
_EDGE_PUNCTUATION = string.punctuation + "“”‘’«»…–—¡¿"
_CLAUSE_END = (",", ".", ";", ":", "?", "!")

#: The article "a" is reduced in connected speech, but espeak-ng gives the letter
#: name eɪ when the word ends a clause or stands alone.
DEFAULT_OVERRIDES: Mapping[str, tuple[str, ...]] = {"a": ("ɐ",)}

Runner = Callable[[str], str]


class EspeakError(RuntimeError):
    """espeak-ng failed to produce phonemes."""


class EspeakNotFoundError(EspeakError):
    """The espeak-ng executable is not installed or not on PATH."""


@dataclass(frozen=True)
class WordPhonemes:
    """A word as written, with its expected phoneme sequence."""

    word: str
    phonemes: tuple[str, ...]


def tokenize(text: str) -> list[str]:
    """Split on whitespace and keep tokens that contain a letter or digit."""
    return [token for token in text.split() if _HAS_ALNUM.search(token)]


def display_form(token: str) -> str:
    """Strip surrounding punctuation for display, keeping inner apostrophes and hyphens."""
    return token.strip(_EDGE_PUNCTUATION) or token


def _cut_point(tokens: list[str]) -> int:
    """Split a token list in two, preferring a clause boundary near the middle."""
    middle = len(tokens) // 2
    candidates = [i + 1 for i, token in enumerate(tokens[:-1]) if token.endswith(_CLAUSE_END)]
    if candidates:
        return min(candidates, key=lambda i: abs(i - middle))
    return middle


class Phonemizer:
    """Turn text into word-aligned phoneme sequences using espeak-ng."""

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        *,
        espeak_path: str = "espeak-ng",
        timeout: float = 15.0,
        runner: Runner | None = None,
        overrides: Mapping[str, tuple[str, ...]] | None = DEFAULT_OVERRIDES,
    ) -> None:
        self.voice = voice
        self.espeak_path = espeak_path
        self.timeout = timeout
        self.overrides = {k.lower(): tuple(v) for k, v in (overrides or {}).items()}
        self._run = runner or self._run_espeak
        self._cache: dict[str, list[list[str]]] = {}

    @staticmethod
    def is_available(espeak_path: str = "espeak-ng") -> bool:
        return shutil.which(espeak_path) is not None

    def phonemize(self, text: str) -> list[WordPhonemes]:
        """Return one entry per word token, in order.

        The whole sentence is phonemised at once so context-dependent forms
        survive ("the" is ðɪ before a vowel and ðə before a consonant). espeak-ng
        sometimes glues unstressed neighbours into one output word ("of the" →
        ʌvðə) or expands a token into several (numbers, abbreviations); when its
        word count differs from ours the tokens are bisected and each half is
        phonemised again until every piece lines up, so context is lost only
        next to the cut.
        """
        tokens = tokenize(text)
        if not tokens:
            return []
        groups = self._phonemize_tokens(tokens)
        words = []
        for token, phonemes in zip(tokens, groups, strict=True):
            word = display_form(token)
            override = self.overrides.get(word.lower())
            words.append(WordPhonemes(word, override if override is not None else tuple(phonemes)))
        return words

    def _phonemize_tokens(self, tokens: list[str]) -> list[list[str]]:
        groups = self._phonemize_chunk(" ".join(tokens))
        if len(groups) == len(tokens):
            return groups
        if len(tokens) == 1:
            return [[p for group in groups for p in group]]
        cut = _cut_point(tokens)
        return self._phonemize_tokens(tokens[:cut]) + self._phonemize_tokens(tokens[cut:])

    def _phonemize_chunk(self, text: str) -> list[list[str]]:
        cached = self._cache.get(text)
        if cached is None:
            if len(self._cache) >= 4096:
                self._cache.clear()
            cached = self._cache[text] = self.parse(self._run(text))
        return [list(group) for group in cached]

    @staticmethod
    def parse(raw: str) -> list[list[str]]:
        """Parse ``espeak-ng --ipa --sep=_`` output into per-word phoneme lists.

        espeak-ng writes one line per clause, words separated by spaces and
        phonemes by underscores. Stress marks ride on the vowel they precede and
        some internal phonemes have no IPA form, which leaves empty fields.
        """
        words: list[list[str]] = []
        for chunk in raw.split():
            if _LANGUAGE_FLAG.match(chunk):
                continue
            phonemes = [p for p in chunk.translate(_DROP_CHARS).split(_SEPARATOR) if p]
            if phonemes:
                words.append(phonemes)
        return words

    def _run_espeak(self, text: str) -> str:
        command = [
            self.espeak_path,
            "-q",
            "--ipa",
            f"--sep={_SEPARATOR}",
            "-v",
            self.voice,
            "--stdin",
        ]
        try:
            # The trailing newline matters: espeak-ng 1.52 mangles a final word
            # that is not terminated by whitespace.
            proc = subprocess.run(
                command,
                input=text + "\n",
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise EspeakNotFoundError(
                f"espeak-ng was not found at {self.espeak_path!r}; install it with "
                "`brew install espeak-ng` or `apt-get install espeak-ng`"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise EspeakError(f"espeak-ng timed out after {self.timeout:g}s") from exc
        if proc.returncode != 0:
            raise EspeakError(f"espeak-ng exited with {proc.returncode}: {proc.stderr.strip()}")
        return proc.stdout
