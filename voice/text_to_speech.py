"""Text-to-speech helpers using the Coqui TTS library."""

import logging
import threading
import re

import numpy as np
import sounddevice as sd
import torch
from TTS.api import TTS

# Determine whether a GPU is available for faster synthesis.
# Prefer the first discrete GPU (usually index 0) when present.
device_count = getattr(torch.cuda, "device_count", lambda: 0)()
DEVICE = "cuda:0" if device_count > 0 else "cpu"

# Load the TTS model only once at module import time and place it on the
# selected device (CPU or GPU).
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
tts.to(DEVICE)


def _chunk_text(text: str, max_chars: int = 300):
    """Yield chunks of ``text`` no longer than ``max_chars``."""
    sentences = re.split(r"(?<=[.!?]) +", text)
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + (1 if current else 0) <= max_chars:
            current = f"{current} {sent}".strip()
        else:
            if current:
                yield current
            if len(sent) <= max_chars:
                current = sent
            else:
                # Fallback: split long sentence directly
                for i in range(0, len(sent), max_chars):
                    yield sent[i : i + max_chars]
                current = ""
    if current:
        yield current

def speak(text: str) -> None:
    """Synthesize ``text`` to speech and play it on a background thread."""

    def _worker(txt: str) -> None:
        logging.info(f"Coqui TTS synthesizing: '{txt}'")
        try:
            for chunk in _chunk_text(txt):
                wav: np.ndarray = tts.tts(chunk)
                sample_rate: int = tts.synthesizer.output_sample_rate
                sd.play(wav, samplerate=sample_rate)
                sd.wait()
        except Exception as e:
            logging.error(f"TTS synthesis failed: {e}", exc_info=True)

    # Offload heavy synthesis and playback so the caller returns immediately.
    threading.Thread(target=_worker, args=(text,), daemon=True).start()

