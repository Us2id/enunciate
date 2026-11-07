# Contributing

Issues and pull requests are welcome.

## Setup

```sh
brew install espeak-ng          # or: sudo apt-get install espeak-ng
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # Linux: avoids the CUDA build
pip install -e ".[dev]"
pre-commit install
```

## Checks

```sh
make lint typecheck test      # ruff, mypy, unit tests (no model needed)
make integration              # real model on the LibriSpeech fixtures
make evaluate                 # phoneme error rate on 73 LibriSpeech clips
```

Unit tests must not need the model or espeak-ng; use the fakes in
`tests/conftest.py`. Tests that need either go under `tests/integration` or are
skipped when espeak-ng is missing.

## Changing the accepted-variant rules

Run `python scripts/evaluate.py --strict` before and `python scripts/evaluate.py`
after, and put the numbers in the pull request. A rule should name a feature of
spoken English or a spelling difference between espeak-ng and the model, not a
learner error.
