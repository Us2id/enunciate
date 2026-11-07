import numpy as np

from enunciate import Audio
from enunciate.cli import build_parser, format_assessment, settings_from
from tests.conftest import make_assessor


def test_format_assessment_lists_words_and_differences():
    assessment = make_assessor(["aɪ", "l", "ʌ", "b", "ɪ", "n", "d", "i", "ə"]).assess(
        Audio(np.zeros(16_000, dtype=np.float32), 16_000), "I love India"
    )
    text = format_assessment(assessment)
    assert text.startswith("Score 0.89: 8 of 9 sounds matched, 1 different")
    assert "love   l ʌ v      l ʌ b      v→b  (0.67)" in text
    assert text.endswith("Heard: aɪ l ʌ b ɪ n d i ə")


def test_format_assessment_shows_accepted_variants_dimly():
    assessment = make_assessor(["ð", "ɐ", "k", "ɑːɹ"]).assess(
        Audio(np.zeros(16_000, dtype=np.float32), 16_000), "the car"
    )
    text = format_assessment(assessment, color=True)
    assert "Score 1.00: 5 of 5 sounds matched" in text
    assert "\033[2m(ə≈ɐ)\033[0m" in text
    assert "accepted variants" in text


def test_format_assessment_does_not_count_accepted_insertions_as_matched():
    assessment = make_assessor(["n", "ɔː", "ɹ", "ɹ"]).assess(
        Audio(np.zeros(16_000, dtype=np.float32), 16_000), "nor"
    )
    text = format_assessment(assessment)
    assert text.startswith("Score 1.00: 3 of 3 sounds matched")
    assert "(+ɹ)" in text


def test_main_reports_missing_files_cleanly(capsys):
    from enunciate.cli import main

    assert main(["assess", "/no/such/file.wav", "hi", "--device", "cpu"]) == 2
    assert "error:" in capsys.readouterr().err


def test_settings_from_arguments_override_environment(monkeypatch):
    monkeypatch.setenv("ENUNCIATE_DEVICE", "cpu")
    args = build_parser().parse_args(["assess", "a.wav", "hi", "--voice", "en-gb", "--strict"])
    settings = settings_from(args)
    assert settings.device == "cpu"
    assert settings.voice == "en-gb"
    assert settings.accept_variants is False
    args = build_parser().parse_args(["serve", "--no-preload", "--model", "org/m"])
    settings = settings_from(args)
    assert settings.preload_model is False
    assert settings.model_id == "org/m"
