"""Runs the real model on real speech. Needs espeak-ng and a one-time model download.

pytest -m integration
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from enunciate import Assessor, Phonemizer, Recognizer, load_audio
from enunciate.api import create_app
from enunciate.config import Settings

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
CLIPS = {
    "1272-128104-0000.flac": (
        "Mister Quilter is the apostle of the middle classes and we are glad to welcome his gospel"
    ),
    "1272-128104-0001.flac": "Nor is Mister Quilter's manner less interesting than his matter",
}
DEVICE = os.environ.get("ENUNCIATE_DEVICE", "auto")


@pytest.fixture(scope="module")
def assessor() -> Assessor:
    instance = Assessor(Recognizer(device=DEVICE), Phonemizer("en-us"))
    instance.warm_up()
    return instance


@pytest.mark.parametrize("clip", sorted(CLIPS))
def test_real_speech_scores_high(assessor: Assessor, clip: str):
    assessment = assessor.assess(load_audio(FIXTURES / clip), CLIPS[clip])
    assert assessment.phoneme_error_rate <= 0.15, assessment.to_json()
    assert assessment.score >= 0.85
    good = sum(w.score >= 0.8 for w in assessment.words) / len(assessment.words)
    assert good >= 0.8
    assert len(assessment.heard) > 0.8 * len(assessment.expected)


def test_wrong_sentence_scores_low(assessor: Assessor):
    audio = load_audio(FIXTURES / "1272-128104-0001.flac")
    assessment = assessor.assess(audio, "The quick brown fox jumps over the lazy dog")
    assert assessment.score < 0.5


def test_silence_yields_no_phonemes(assessor: Assessor):
    import numpy as np

    from enunciate import Audio

    assessment = assessor.assess(Audio(np.zeros(16_000, dtype=np.float32), 16_000), "hello")
    assert assessment.heard == ()
    assert assessment.score == 0.0


def test_cli_json_output():
    clip = "1272-128104-0001.flac"
    result = subprocess.run(
        [sys.executable, "-m", "enunciate", "assess", str(FIXTURES / clip), CLIPS[clip], "--json"],
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
        env={**os.environ, "ENUNCIATE_DEVICE": DEVICE},
    )
    body = json.loads(result.stdout)
    assert body["score"] >= 0.85
    assert [w["word"] for w in body["words"]][:3] == ["Nor", "is", "Mister"]


def test_api_end_to_end():
    settings = Settings(device=DEVICE, preload_model=True)
    with TestClient(create_app(settings)) as client:
        assert client.get("/healthz").json()["model_loaded"] is True
        clip = "1272-128104-0001.flac"
        response = client.post(
            "/api/assess",
            data={"text": CLIPS[clip]},
            files={"audio": (clip, (FIXTURES / clip).read_bytes(), "audio/flac")},
        )
    assert response.status_code == 200, response.text
    assert response.json()["score"] >= 0.85
