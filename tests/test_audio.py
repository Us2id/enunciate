import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from enunciate.audio import Audio, AudioDecodeError, from_array, load_audio


def sine(rate: int, seconds: float, channels: int = 1, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    wave = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    return np.stack([wave] * channels, axis=1) if channels > 1 else wave


def encode(samples: np.ndarray, rate: int, fmt: str = "WAV", subtype: str | None = None) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, samples, rate, format=fmt, subtype=subtype)
    return buffer.getvalue()


def test_wav_stereo_44k_becomes_mono_16k():
    audio = load_audio(encode(sine(44_100, 1.0, channels=2), 44_100))
    assert audio.sample_rate == 16_000
    assert audio.samples.dtype == np.float32
    assert audio.samples.ndim == 1
    assert audio.duration == pytest.approx(1.0, abs=0.01)
    assert np.abs(audio.samples).max() == pytest.approx(0.5, abs=0.02)


def test_path_bytes_and_file_object_inputs(tmp_path: Path):
    data = encode(sine(16_000, 0.5), 16_000)
    path = tmp_path / "clip.wav"
    path.write_bytes(data)
    from_path = load_audio(path)
    from_str = load_audio(str(path))
    from_bytes = load_audio(data)
    with path.open("rb") as handle:
        from_file = load_audio(handle)
    for audio in (from_path, from_str, from_bytes, from_file):
        assert audio.duration == pytest.approx(0.5, abs=0.001)
        np.testing.assert_array_equal(audio.samples, from_path.samples)


def test_flac_and_16bit_pcm_decode():
    assert load_audio(encode(sine(22_050, 0.3), 22_050, fmt="FLAC")).duration == pytest.approx(
        0.3, abs=0.01
    )
    audio = load_audio(encode(sine(16_000, 0.3), 16_000, subtype="PCM_16"))
    assert audio.samples.dtype == np.float32
    assert np.abs(audio.samples).max() <= 1.0


def test_garbage_and_empty_input_raise():
    with pytest.raises(AudioDecodeError):
        load_audio(b"this is not audio at all, just text " * 50)
    with pytest.raises(AudioDecodeError):
        load_audio(b"")


def test_from_array_accepts_int16_and_multichannel():
    pcm = (sine(8_000, 0.25, channels=2) * 32767).astype(np.int16)
    audio = from_array(pcm.astype(np.float32) / 32768.0, 8_000)
    assert audio.sample_rate == 16_000
    assert audio.duration == pytest.approx(0.25, abs=0.01)


def test_audio_validation():
    with pytest.raises(ValueError):
        Audio(np.zeros((10, 2), dtype=np.float32), 16_000)
    with pytest.raises(ValueError):
        Audio(np.zeros(10, dtype=np.int16), 16_000)
    with pytest.raises(ValueError):
        Audio(np.zeros(10, dtype=np.float32), 0)
