"""Phoneme recognition with a CTC-trained wav2vec2 model.

The default model, ``facebook/wav2vec2-lv-60-espeak-cv-ft``, emits espeak-ng
IPA symbols directly from audio. There is no language model on top, so the
output follows what was said rather than what would be the likeliest sentence.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio import TARGET_SAMPLE_RATE, Audio

DEFAULT_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
SPECIAL_TOKENS = frozenset({"<pad>", "<s>", "</s>", "<unk>"})

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeardPhoneme:
    """A recognised phoneme with its time span in seconds."""

    symbol: str
    start: float
    end: float


def resolve_device(requested: str = "auto") -> str:
    """Map ``auto`` to the best available torch device."""
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Recognizer:
    """Lazy-loading, thread-safe wrapper around a wav2vec2 CTC phoneme model."""

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "auto") -> None:
        self.model_id = model_id
        self.requested_device = device
        self.device: str | None = None
        self._lock = threading.Lock()
        # transformers objects are typed loosely upstream; keep them as Any.
        self._model: Any = None
        self._extractor: Any = None
        self._tokens: list[str] = []
        self._blank_id = 0
        self._frame_stride = 320

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def vocabulary(self) -> list[str]:
        """Output symbols in id order (loads the model if needed)."""
        self.load()
        return list(self._tokens)

    def load(self) -> None:
        """Download (if needed) and load the model. Safe to call repeatedly."""
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

            device = resolve_device(self.requested_device)
            started = time.perf_counter()
            log.info("loading %s on %s", self.model_id, device)
            model: Any = Wav2Vec2ForCTC.from_pretrained(self.model_id)
            model.to(device)
            model.eval()
            torch.set_grad_enabled(False)
            self._extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_id)
            self._tokens = _load_vocabulary(self.model_id)
            pad_token_id = model.config.pad_token_id
            self._blank_id = int(pad_token_id) if pad_token_id is not None else 0
            self._frame_stride = math.prod(model.config.conv_stride)
            self._model = model
            self.device = device
            log.info("model ready in %.1fs", time.perf_counter() - started)

    def warm_up(self) -> None:
        """Load the model and run one short forward pass so the first real call is fast."""
        self.load()
        import numpy as np

        self.recognize(Audio(np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32), TARGET_SAMPLE_RATE))

    def recognize(self, audio: Audio) -> list[HeardPhoneme]:
        """Greedy CTC decoding of one utterance."""
        if audio.sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError(f"audio must be {TARGET_SAMPLE_RATE} Hz, got {audio.sample_rate}")
        self.load()
        assert self._model is not None and self._extractor is not None
        import torch

        features = self._extractor(
            audio.samples, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
        )
        input_values = features.input_values.to(self.device)
        attention_mask = features.attention_mask.to(self.device)
        with self._lock, torch.inference_mode():
            logits = self._model(input_values, attention_mask=attention_mask).logits
        ids: list[int] = logits[0].argmax(dim=-1).cpu().tolist()
        return self._collapse(ids)

    def _collapse(self, ids: Sequence[int]) -> list[HeardPhoneme]:
        """Collapse repeated frames and drop blanks, keeping frame boundaries."""
        seconds_per_frame = self._frame_stride / TARGET_SAMPLE_RATE
        heard: list[HeardPhoneme] = []
        previous: int | None = None
        start = 0
        for frame, token_id in enumerate([*ids, self._blank_id]):
            if token_id == previous:
                continue
            if previous is not None and previous != self._blank_id:
                symbol = self._tokens[previous]
                if symbol not in SPECIAL_TOKENS:
                    heard.append(
                        HeardPhoneme(
                            symbol=symbol,
                            start=round(start * seconds_per_frame, 3),
                            end=round(frame * seconds_per_frame, 3),
                        )
                    )
            previous = token_id
            start = frame
        return heard


def _load_vocabulary(model_id: str) -> list[str]:
    local = Path(model_id) / "vocab.json"
    if local.is_file():
        path = local
    else:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(repo_id=model_id, filename="vocab.json"))
    vocab: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    tokens = [""] * (max(int(i) for i in vocab.values()) + 1)
    for symbol, index in vocab.items():
        tokens[int(index)] = symbol
    return tokens
