# Changelog

## 0.1.0 (2026-11-07)

First release.

- Phoneme recognition with `facebook/wav2vec2-lv-60-espeak-cv-ft`, greedy CTC
  decoding with per-phoneme timing.
- Reference pronunciation from espeak-ng with sentence-level context, a
  bisecting fallback for merged or expanded words, and the article "a" fixed
  to its reduced form.
- Levenshtein alignment with pluggable costs; accepted-variant rules for
  General American English derived from LibriSpeech confusions; strict mode.
- Per-word and overall scores with every aligned phoneme reported.
- Command line (`assess`, `phonemes`, `serve`), FastAPI service with limits and
  health endpoint, browser demo that records and converts audio client-side.
- Unit tests with fakes, integration tests on LibriSpeech clips, GitHub Actions
  CI, Dockerfile, evaluation script.
