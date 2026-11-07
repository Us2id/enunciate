"""Measure recogniser/G2P agreement on the LibriSpeech dummy split.

The split (73 clean read-speech utterances from LibriSpeech dev-clean, CC BY 4.0)
is fetched through the Hugging Face datasets server, so no dataset library is
needed. For each clip the transcript is phonemised with espeak-ng and compared
with what the model heard, exactly as an assessment does. The resulting phoneme
error rate is an upper bound on recogniser error: it also counts legitimate
pronunciation variants that differ from espeak's dictionary form.

Usage: python scripts/evaluate.py [--limit N] [--device cpu|mps|cuda] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

from enunciate import ENGLISH_RULES, STRICT_RULES, Assessor, Phonemizer, Recognizer, load_audio

ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=hf-internal-testing/librispeech_asr_dummy&config=clean&split=validation"
    "&offset={offset}&length=100"
)


def fetch_rows(limit: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    offset = 0
    while True:
        with urllib.request.urlopen(ROWS_URL.format(offset=offset), timeout=60) as response:
            page = json.load(response)
        for item in page["rows"]:
            row = item["row"]
            rows.append({"id": row["id"], "text": row["text"], "url": row["audio"][0]["src"]})
            if limit and len(rows) >= limit:
                return rows
        offset += len(page["rows"])
        if offset >= page["num_rows_total"] or not page["rows"]:
            return rows


def download(url: str, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        target.write_bytes(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--limit", type=int, default=None, help="number of clips (default: all 73)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--voice", default="en-us")
    parser.add_argument("--strict", action="store_true", help="count accepted variants as errors")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "enunciate" / "librispeech_dummy",
    )
    parser.add_argument("--json", type=Path, default=None, help="write per-clip results here")
    args = parser.parse_args()

    rows = fetch_rows(args.limit)
    print(f"{len(rows)} clips", file=sys.stderr)
    rules = STRICT_RULES if args.strict else ENGLISH_RULES
    assessor = Assessor(
        Recognizer(device=args.device), Phonemizer(args.voice), rules=rules, max_audio_seconds=60
    )
    assessor.warm_up()

    confusions: Counter[tuple[str, str]] = Counter()
    totals = Counter()
    word_scores: list[float] = []
    results = []
    inference_seconds = 0.0
    audio_seconds = 0.0
    for row in rows:
        path = args.cache_dir / f"{row['id']}.flac"
        download(row["url"], path)
        audio = load_audio(path)
        started = time.perf_counter()
        assessment = assessor.assess(audio, row["text"].lower())
        inference_seconds += time.perf_counter() - started
        audio_seconds += audio.duration
        totals.update(
            expected=len(assessment.expected),
            matches=assessment.matches,
            accepted=assessment.accepted,
            substitutions=assessment.substitutions,
            deletions=assessment.deletions,
            insertions=assessment.insertions,
        )
        for word in assessment.words:
            word_scores.append(word.score)
            for phoneme in word.phonemes:
                if not phoneme.accepted:
                    confusions[(phoneme.expected or "∅", phoneme.heard or "∅")] += 1
        results.append(
            {
                "id": row["id"],
                "text": row["text"],
                "score": assessment.score,
                "per": assessment.phoneme_error_rate,
            }
        )
        per = assessment.phoneme_error_rate
        print(f"{row['id']}  PER={per:.3f}  {row['text'][:60]}", file=sys.stderr)

    n = totals["expected"]
    errors = totals["substitutions"] + totals["deletions"] + totals["insertions"]
    speed = inference_seconds / audio_seconds
    print()
    print(
        f"clips: {len(rows)}   audio: {audio_seconds:.0f}s   inference: {inference_seconds:.1f}s "
        f"({speed:.3f}x real time on {assessor.recognizer.device})"
    )
    print(f"expected phonemes: {n}   accepted variants: {totals['accepted']}")
    print(
        f"PER: {errors / n:.3f}  (sub {totals['substitutions'] / n:.3f}, "
        f"del {totals['deletions'] / n:.3f}, ins {totals['insertions'] / n:.3f})"
    )
    good = sum(s >= 0.8 for s in word_scores) / len(word_scores)
    perfect = sum(s == 1.0 for s in word_scores) / len(word_scores)
    print(
        f"words: {len(word_scores)}   mean word score: {sum(word_scores) / len(word_scores):.3f}   "
        f"words >= 0.8: {good:.3f}   words == 1.0: {perfect:.3f}"
    )
    print("most common confusions (expected -> heard):")
    for (expected, heard), count in confusions.most_common(25):
        print(f"  {expected:>5} -> {heard:<5} {count}")
    if args.json:
        report = {
            "clips": results,
            "totals": dict(totals),
            "confusions": [
                {"expected": e, "heard": h, "count": c} for (e, h), c in confusions.most_common()
            ],
        }
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
