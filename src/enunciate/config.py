"""Runtime settings, overridable through ``ENUNCIATE_*`` environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from .g2p import DEFAULT_VOICE
from .recognizer import DEFAULT_MODEL

ENV_PREFIX = "ENUNCIATE_"
_TRUE = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True)
class Settings:
    model_id: str = DEFAULT_MODEL
    device: str = "auto"
    voice: str = DEFAULT_VOICE
    espeak_path: str = "espeak-ng"
    max_audio_seconds: float = 30.0
    min_audio_seconds: float = 0.2
    max_upload_bytes: int = 10 * 1024 * 1024
    max_text_chars: int = 500
    cors_origins: tuple[str, ...] = ()
    preload_model: bool = True
    accept_variants: bool = True

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from the environment (``ENUNCIATE_MODEL``, ``ENUNCIATE_DEVICE``, ...)."""
        source = os.environ if env is None else env
        defaults = cls()

        def read(name: str, default: str) -> str:
            return source.get(ENV_PREFIX + name, default)

        origins = tuple(o.strip() for o in read("CORS_ORIGINS", "").split(",") if o.strip())
        return cls(
            model_id=read("MODEL", defaults.model_id),
            device=read("DEVICE", defaults.device),
            voice=read("VOICE", defaults.voice),
            espeak_path=read("ESPEAK", defaults.espeak_path),
            max_audio_seconds=float(read("MAX_AUDIO_SECONDS", str(defaults.max_audio_seconds))),
            min_audio_seconds=float(read("MIN_AUDIO_SECONDS", str(defaults.min_audio_seconds))),
            max_upload_bytes=int(float(read("MAX_UPLOAD_MB", "10")) * 1024 * 1024),
            max_text_chars=int(read("MAX_TEXT_CHARS", str(defaults.max_text_chars))),
            cors_origins=origins,
            preload_model=read("PRELOAD_MODEL", "1").strip().lower() in _TRUE,
            accept_variants=read("ACCEPT_VARIANTS", "1").strip().lower() in _TRUE,
        )
