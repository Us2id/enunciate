"""Audio decoding and resampling.

Everything downstream works on mono float32 samples at 16 kHz, which is what
wav2vec2 models expect. ``soundfile`` (libsndfile) decodes WAV, FLAC, OGG and
MP3 natively; anything else is handed to ``ffmpeg`` when it is installed.
"""

from __future__ import annotations

import io
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import soundfile as sf
import soxr

TARGET_SAMPLE_RATE = 16_000


class AudioDecodeError(ValueError):
    """The input could not be decoded as audio."""


@dataclass(frozen=True)
class Audio:
    """Mono float32 samples in the range [-1, 1]."""

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        if self.samples.ndim != 1:
            raise ValueError(f"expected mono samples with shape (n,), got {self.samples.shape}")
        if self.samples.dtype != np.float32:
            raise ValueError(f"expected float32 samples, got {self.samples.dtype}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample rate must be positive, got {self.sample_rate}")

    @property
    def duration(self) -> float:
        """Length in seconds."""
        return len(self.samples) / self.sample_rate


def load_audio(
    source: str | Path | bytes | BinaryIO,
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> Audio:
    """Decode a file path, raw bytes or a binary file object into :class:`Audio`.

    Multi-channel input is averaged to mono and resampled to ``sample_rate``.
    """
    data = _read_bytes(source)
    if not data:
        raise AudioDecodeError("audio input is empty")
    try:
        samples, native_rate = _decode_with_soundfile(data)
    except (sf.LibsndfileError, RuntimeError, ValueError) as exc:
        if shutil.which("ffmpeg") is None:
            raise AudioDecodeError(
                "could not decode audio; supported formats are WAV, FLAC, OGG and MP3 "
                "(install ffmpeg for other containers)"
            ) from exc
        samples, native_rate = _decode_with_ffmpeg(data, sample_rate)
    return from_array(samples, native_rate, sample_rate=sample_rate)


def from_array(
    samples: np.ndarray,
    native_rate: int,
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> Audio:
    """Normalise an array of samples (any dtype, mono or multi-channel) to :class:`Audio`."""
    mono = np.asarray(samples, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    elif mono.ndim != 1:
        raise ValueError(f"unsupported sample array shape {mono.shape}")
    if native_rate != sample_rate:
        mono = soxr.resample(mono, native_rate, sample_rate, quality="HQ")
    mono = np.nan_to_num(np.ascontiguousarray(mono, dtype=np.float32))
    return Audio(mono, sample_rate)


def _read_bytes(source: str | Path | bytes | BinaryIO) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, str | Path):
        return Path(source).read_bytes()
    return source.read()


def _decode_with_soundfile(data: bytes) -> tuple[np.ndarray, int]:
    samples, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return samples, int(rate)


def _decode_with_ffmpeg(data: bytes, sample_rate: int) -> tuple[np.ndarray, int]:
    command = [
        "ffmpeg",
        "-v", "error",
        "-i", "pipe:0",
        "-f", "f32le",
        "-ac", "1",
        "-ar", str(sample_rate),
        "pipe:1",
    ]  # fmt: skip
    try:
        proc = subprocess.run(command, input=data, capture_output=True, timeout=120, check=False)
    except subprocess.TimeoutExpired as exc:
        raise AudioDecodeError("ffmpeg timed out while decoding audio") from exc
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", errors="replace").strip().splitlines()
        reason = detail[-1] if detail else "no output"
        raise AudioDecodeError(f"ffmpeg could not decode audio: {reason}")
    return np.frombuffer(proc.stdout, dtype=np.float32), sample_rate
