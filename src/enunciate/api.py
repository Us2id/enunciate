"""HTTP API and browser demo, built with FastAPI.

Start it with ``enunciate serve`` or mount :func:`create_app` under any ASGI
server. Model inference runs in a worker thread so the event loop stays
responsive, and the recogniser serialises calls to the model, so one process
scores one recording at a time; run several processes to scale out.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from . import __version__
from .assess import AssessmentError, Assessor
from .audio import AudioDecodeError, load_audio
from .config import Settings
from .g2p import SUPPORTED_VOICES, EspeakError, Phonemizer

STATIC_DIR = Path(__file__).parent / "static"
_CHUNK = 64 * 1024

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, assessor: Assessor | None = None) -> FastAPI:
    """Build the application. Tests pass an ``assessor`` with fake components."""
    settings = settings or Settings.from_env()
    assessor = assessor or Assessor.from_settings(settings)
    assessors: dict[str, Assessor] = {assessor.phonemizer.voice: assessor}

    def assessor_for(voice: str | None) -> Assessor:
        """One assessor per reference accent; they share the loaded model."""
        if voice is None:
            return assessor
        if voice not in SUPPORTED_VOICES:
            raise HTTPException(
                422, f"unsupported voice {voice!r}; choose one of {', '.join(SUPPORTED_VOICES)}"
            )
        if voice not in assessors:
            assessors[voice] = Assessor(
                assessor.recognizer,
                Phonemizer(voice, espeak_path=assessor.phonemizer.espeak_path),
                rules=assessor.rules,
                max_audio_seconds=assessor.max_audio_seconds,
                min_audio_seconds=assessor.min_audio_seconds,
                max_text_chars=assessor.max_text_chars,
            )
        return assessors[voice]

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if settings.preload_model:
            await run_in_threadpool(assessor.warm_up)
        yield

    app = FastAPI(
        title="enunciate",
        version=__version__,
        description="Phoneme-level English pronunciation assessment.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.assessor = assessor
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    @app.exception_handler(AssessmentError)
    async def assessment_error(_: Request, exc: AssessmentError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(AudioDecodeError)
    async def audio_error(_: Request, exc: AudioDecodeError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=415)

    @app.exception_handler(EspeakError)
    async def espeak_error(_: Request, exc: EspeakError) -> JSONResponse:
        log.error("espeak-ng failed: %s", exc)
        return JSONResponse({"detail": f"pronunciation dictionary unavailable: {exc}"}, 503)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        """Liveness and readiness in one: ``model_loaded`` flips once the model is warm."""
        return {
            "status": "ok",
            "version": __version__,
            "model": assessor.recognizer.model_id,
            "model_loaded": assessor.recognizer.is_loaded,
            "device": assessor.recognizer.device,
            "voices": list(SUPPORTED_VOICES),
            "limits": {
                "max_audio_seconds": assessor.max_audio_seconds,
                "max_upload_bytes": settings.max_upload_bytes,
                "max_text_chars": assessor.max_text_chars,
            },
        }

    @app.get("/api/phonemes")
    async def phonemes(text: str, voice: str | None = None) -> dict[str, Any]:
        """Expected pronunciation of ``text``, word by word."""
        target = assessor_for(voice)
        words = await run_in_threadpool(target.expected_phonemes, text)
        return {
            "text": text,
            "voice": target.phonemizer.voice,
            "words": [{"word": w.word, "phonemes": list(w.phonemes)} for w in words],
        }

    @app.post("/api/assess")
    async def assess(
        text: Annotated[str, Form()],
        audio: Annotated[UploadFile, File()],
        voice: Annotated[str | None, Form()] = None,
    ) -> dict[str, Any]:
        """Score a recording of ``text``. Send WAV, FLAC, OGG or MP3 as ``audio``."""
        target = assessor_for(voice)
        data = await read_limited(audio, settings.max_upload_bytes)
        decoded = await run_in_threadpool(load_audio, data)
        started = time.perf_counter()
        assessment = await run_in_threadpool(target.assess, decoded, text)
        log.info(
            "assessed %d words in %.1fs of audio: score %.2f (%.0f ms)",
            len(assessment.words),
            assessment.audio_seconds,
            assessment.score,
            (time.perf_counter() - started) * 1000,
        )
        return assessment.to_dict()

    return app


async def read_limited(upload: UploadFile, limit: int) -> bytes:
    """Read an upload, refusing anything above ``limit`` bytes before it is buffered."""
    chunks: list[bytes] = []
    size = 0
    while chunk := await upload.read(_CHUNK):
        size += len(chunk)
        if size > limit:
            raise HTTPException(413, f"audio upload is larger than {limit / (1024 * 1024):g} MB")
        chunks.append(chunk)
    return b"".join(chunks)
