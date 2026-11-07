from enunciate.config import Settings


def test_defaults():
    settings = Settings()
    assert settings.model_id == "facebook/wav2vec2-lv-60-espeak-cv-ft"
    assert settings.device == "auto"
    assert settings.voice == "en-us"
    assert settings.max_upload_bytes == 10 * 1024 * 1024
    assert settings.accept_variants is True


def test_from_env_overrides():
    settings = Settings.from_env(
        {
            "ENUNCIATE_MODEL": "org/model",
            "ENUNCIATE_DEVICE": "cpu",
            "ENUNCIATE_VOICE": "en-gb",
            "ENUNCIATE_MAX_AUDIO_SECONDS": "12.5",
            "ENUNCIATE_MAX_UPLOAD_MB": "2",
            "ENUNCIATE_CORS_ORIGINS": "https://a.example, https://b.example",
            "ENUNCIATE_PRELOAD_MODEL": "false",
            "ENUNCIATE_ACCEPT_VARIANTS": "0",
        }
    )
    assert settings.model_id == "org/model"
    assert settings.device == "cpu"
    assert settings.voice == "en-gb"
    assert settings.max_audio_seconds == 12.5
    assert settings.max_upload_bytes == 2 * 1024 * 1024
    assert settings.cors_origins == ("https://a.example", "https://b.example")
    assert settings.preload_model is False
    assert settings.accept_variants is False


def test_from_env_ignores_unrelated_variables():
    assert Settings.from_env({"HOME": "/tmp"}) == Settings()
