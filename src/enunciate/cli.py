"""Command-line interface: ``enunciate assess``, ``enunciate phonemes``, ``enunciate serve``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace

from . import __version__
from .assess import Assessment, AssessmentError, Assessor
from .audio import AudioDecodeError, load_audio
from .config import Settings
from .g2p import SUPPORTED_VOICES, EspeakError

GOOD = 0.8
PARTIAL = 0.5

_RESET = "\033[0m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_DIM = "\033[2m"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    quiet_libraries()
    try:
        result: int = args.func(args)
    except (AssessmentError, AudioDecodeError, EspeakError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="enunciate",
        description="Phoneme-level English pronunciation assessment.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--log-level", default="warning", help="debug, info, warning or error")
    commands = parser.add_subparsers(dest="command", required=True)

    assess = commands.add_parser(
        "assess", parents=[common], help="score a recording against the sentence it says"
    )
    assess.add_argument("audio", help="WAV, FLAC, OGG or MP3 file (anything else needs ffmpeg)")
    assess.add_argument("text", help="the sentence the speaker meant to say")
    _model_options(assess)
    assess.add_argument("--strict", action="store_true", help="count accepted variants as errors")
    assess.add_argument("--json", action="store_true", help="print the full result as JSON")
    assess.add_argument("--max-seconds", type=float, default=None, help="reject longer audio")
    assess.set_defaults(func=run_assess)

    phonemes = commands.add_parser(
        "phonemes", parents=[common], help="show the expected pronunciation of a text"
    )
    phonemes.add_argument("text")
    phonemes.add_argument("--voice", choices=SUPPORTED_VOICES, default=None)
    phonemes.add_argument("--json", action="store_true")
    phonemes.set_defaults(func=run_phonemes)

    serve = commands.add_parser(
        "serve", parents=[common], help="start the HTTP API and browser demo"
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--no-preload", action="store_true", help="load the model on first request")
    _model_options(serve)
    serve.set_defaults(func=run_serve)
    return parser


def _model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--voice", choices=SUPPORTED_VOICES, default=None, help="reference accent")
    parser.add_argument("--model", default=None, help="Hugging Face model id or local path")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda or mps")


def settings_from(args: argparse.Namespace) -> Settings:
    """Environment settings with command-line overrides applied."""
    settings = Settings.from_env()
    overrides = {
        key: value
        for key, value in {
            "voice": getattr(args, "voice", None),
            "model_id": getattr(args, "model", None),
            "device": getattr(args, "device", None),
            "max_audio_seconds": getattr(args, "max_seconds", None),
        }.items()
        if value is not None
    }
    if getattr(args, "strict", False):
        overrides["accept_variants"] = False
    if getattr(args, "no_preload", False):
        overrides["preload_model"] = False
    return replace(settings, **overrides)


def run_assess(args: argparse.Namespace) -> int:
    settings = settings_from(args)
    assessor = Assessor.from_settings(settings)
    audio = load_audio(args.audio)
    assessment = assessor.assess(audio, args.text)
    if args.json:
        print(assessment.to_json(indent=2))
    else:
        print(format_assessment(assessment, color=sys.stdout.isatty()))
    return 0


def run_phonemes(args: argparse.Namespace) -> int:
    settings = settings_from(args)
    words = Assessor.from_settings(settings).expected_phonemes(args.text)
    if args.json:
        import json

        print(
            json.dumps(
                [{"word": w.word, "phonemes": list(w.phonemes)} for w in words], ensure_ascii=False
            )
        )
        return 0
    width = max((len(w.word) for w in words), default=0)
    for word in words:
        print(f"{word.word:<{width}}  {' '.join(word.phonemes)}")
    return 0


def run_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:  # pragma: no cover - depends on the installed extras
        print(
            "error: the server needs the extra dependencies: pip install 'enunciate[server]'",
            file=sys.stderr,
        )
        return 2
    from .api import create_app

    settings = settings_from(args)
    uvicorn.run(
        create_app(settings), host=args.host, port=args.port, log_level=args.log_level.lower()
    )
    return 0


def format_assessment(assessment: Assessment, *, color: bool = False) -> str:
    """Human-readable table: one line per word, differences spelled out."""

    def paint(text: str, code: str) -> str:
        return f"{code}{text}{_RESET}" if color else text

    matched = len(assessment.expected) - assessment.substitutions - assessment.deletions
    errors = assessment.substitutions + assessment.deletions + assessment.insertions
    headline = (
        f"Score {assessment.score:.2f}: {matched} of {len(assessment.expected)} sounds matched"
    )
    if errors:
        parts = [
            f"{count} {name}"
            for count, name in (
                (assessment.substitutions, "different"),
                (assessment.deletions, "missing"),
                (assessment.insertions, "extra"),
            )
            if count
        ]
        headline += f", {', '.join(parts)}"
    lines = [headline, ""]
    width = max(len(w.word) for w in assessment.words)
    expected_width = max(len(" ".join(w.expected)) for w in assessment.words)
    heard_width = max(len(" ".join(w.heard)) for w in assessment.words)
    any_accepted = False
    for word in assessment.words:
        notes = []
        for p in word.phonemes:
            if p.op == "match":
                continue
            if p.accepted:
                any_accepted = True
                if p.op == "substitute":
                    note = f"{p.expected}≈{p.heard}"
                elif p.op == "insert":
                    note = f"+{p.heard}"
                else:
                    note = f"{p.expected} dropped"
                notes.append(paint(f"({note})", _DIM))
            elif p.op == "substitute":
                notes.append(f"{p.expected}→{p.heard}")
            elif p.op == "delete":
                notes.append(f"{p.expected} missing")
            else:
                notes.append(f"+{p.heard}")
        line = (
            f"  {word.word:<{width}}  {' '.join(word.expected):<{expected_width}}  "
            f"{' '.join(word.heard):<{heard_width}}  {' '.join(notes)}"
        ).rstrip()
        if word.score < 1.0:
            line += f"  ({word.score:.2f})"
        if word.score < PARTIAL:
            line = paint(line, _RED)
        elif word.score < GOOD:
            line = paint(line, _YELLOW)
        lines.append(line)
    lines.append("")
    lines.append(f"Heard: {' '.join(h.symbol for h in assessment.heard) or '(nothing)'}")
    if any_accepted:
        lines.append(paint("Differences in parentheses are accepted variants.", _DIM))
    return "\n".join(lines)


def quiet_libraries() -> None:
    """Hide progress bars and advisory notices from the model libraries."""
    try:
        from transformers.utils import logging as hf_logging
    except ImportError:  # pragma: no cover
        return
    disable_progress_bar = getattr(hf_logging, "disable_progress_bar", None)
    if disable_progress_bar is not None:
        disable_progress_bar()
    for name in ("huggingface_hub", "httpx", "httpx2"):
        logging.getLogger(name).setLevel(logging.WARNING)
    # torch.distributed logs a macOS note about output redirects on import.
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
