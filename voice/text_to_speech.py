"""Text-to-speech helpers using the Coqui TTS library."""

import logging
import threading

import numpy as np
import sounddevice as sd
import torch
from TTS.api import TTS

# Determine whether a GPU is available for faster synthesis.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the TTS model only once at module import time and place it on the
# selected device (CPU or GPU).
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
tts.to(DEVICE)


def speak(text: str) -> None:
    """Synthesize ``text`` to speech and play it on a background thread."""
    logging.info(f"Coqui TTS synthesizing: '{text}'")
    try:
        wav: np.ndarray = tts.tts(text)
        sample_rate: int = tts.synthesizer.output_sample_rate

        def _playback(data: np.ndarray, sr: int) -> None:
            """Play the generated audio and block until finished."""
            sd.play(data, samplerate=sr)
            sd.wait()

        # Playing audio can block, so spawn a daemon thread to keep the UI
        # responsive.
        threading.Thread(
            target=_playback, args=(wav, sample_rate), daemon=True
        ).start()
    except Exception as e:
        logging.error(f"TTS synthesis failed: {e}", exc_info=True)

