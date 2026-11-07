import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from enunciate.api import create_app
from enunciate.config import Settings
from enunciate.g2p import Phonemizer
from tests.conftest import make_assessor

HEARD = ["aɪ", "l", "ʌ", "b", "ɪ", "n", "d", "i", "ə"]


def wav_bytes(seconds: float = 1.0, rate: int = 16_000) -> bytes:
    buffer = io.BytesIO()
    t = np.arange(int(rate * seconds)) / rate
    sf.write(buffer, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), rate, format="WAV")
    return buffer.getvalue()


@pytest.fixture
def client() -> TestClient:
    settings = Settings(preload_model=False, max_upload_bytes=200_000, max_audio_seconds=5.0)
    assessor = make_assessor(HEARD, max_audio_seconds=5.0)
    with TestClient(create_app(settings, assessor)) as client:
        yield client


def test_healthz_before_and_after_preload():
    settings = Settings(preload_model=True)
    with TestClient(create_app(settings, make_assessor(HEARD))) as client:
        body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model"] == "fake/model"
    assert body["voices"] == ["en-us", "en-gb"]


def test_index_serves_the_demo_page(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<title>" in response.text


def test_phonemes_endpoint(client: TestClient):
    body = client.get("/api/phonemes", params={"text": "I love India"}).json()
    assert body["voice"] == "en-us"
    assert body["words"] == [
        {"word": "I", "phonemes": ["aɪ"]},
        {"word": "love", "phonemes": ["l", "ʌ", "v"]},
        {"word": "India", "phonemes": ["ɪ", "n", "d", "i", "ə"]},
    ]


def test_phonemes_rejects_empty_text(client: TestClient):
    assert client.get("/api/phonemes", params={"text": "   "}).status_code == 422
    assert client.get("/api/phonemes").status_code == 422


def test_assess_returns_scores(client: TestClient):
    response = client.post(
        "/api/assess",
        data={"text": "I love India"},
        files={"audio": ("clip.wav", wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["text"] == "I love India"
    assert [w["word"] for w in body["words"]] == ["I", "love", "India"]
    assert body["words"][1]["substitutions"] == 1
    assert body["words"][1]["phonemes"][2] == {
        "op": "substitute",
        "expected": "v",
        "heard": "b",
        "accepted": False,
        "start": pytest.approx(0.333, abs=0.001),
        "end": pytest.approx(0.444, abs=0.001),
    }
    assert body["score"] == pytest.approx(8 / 9, abs=1e-3)
    assert body["audio_seconds"] == 1.0


def test_assess_validation_errors(client: TestClient):
    wav = wav_bytes()
    too_long = client.post(
        "/api/assess", data={"text": "I"}, files={"audio": ("c.wav", wav_bytes(6.0), "audio/wav")}
    )
    assert too_long.status_code == 422
    assert "limit is 5" in too_long.json()["detail"]

    empty = client.post(
        "/api/assess", data={"text": " "}, files={"audio": ("c.wav", wav, "audio/wav")}
    )
    assert empty.status_code == 422

    garbage = client.post(
        "/api/assess", data={"text": "I"}, files={"audio": ("c.txt", b"nope" * 100, "text/plain")}
    )
    assert garbage.status_code == 415

    huge = client.post(
        "/api/assess", data={"text": "I"}, files={"audio": ("c.wav", b"\0" * 300_000, "audio/wav")}
    )
    assert huge.status_code == 413

    missing = client.post("/api/assess", data={"text": "I"})
    assert missing.status_code == 422

    voice = client.post(
        "/api/assess",
        data={"text": "I", "voice": "fr-fr"},
        files={"audio": ("c.wav", wav, "audio/wav")},
    )
    assert voice.status_code == 422


@pytest.mark.skipif(not Phonemizer.is_available(), reason="espeak-ng not installed")
def test_other_voice_uses_its_own_phonemizer(client: TestClient):
    body = client.get("/api/phonemes", params={"text": "water", "voice": "en-gb"}).json()
    assert body["voice"] == "en-gb"
    assert body["words"][0]["phonemes"] == ["w", "ɔː", "t", "ə"]
