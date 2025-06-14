# voice/text_to_speech.py

import logging
import threading

import torch
import numpy as np
import sounddevice as sd
from TTS.api import TTS

# Detect device once
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model once and move it to the chosen device
tts = TTS(
    model_name="tts_models/en/ljspeech/tacotron2-DDC",
    progress_bar=False
)
tts.to(DEVICE)  # replace gpu=... usage with explicit .to(device) per deprecation warning

def speak(text: str):
    """
    Synthesize `text` to speech and play it in a background thread.
    """
    logging.info(f"Coqui TTS synthesizing: '{text}'")
    try:
        # Generate waveform on the selected device
        wav: np.ndarray = tts.tts(text)
        # Get the sample rate
        sample_rate: int = tts.synthesizer.output_sample_rate

        # Play back in a separate thread to avoid blocking
        def _playback(data: np.ndarray, sr: int):
            sd.play(data, samplerate=sr)
            sd.wait()

        threading.Thread(
            target=_playback,
            args=(wav, sample_rate),
            daemon=True
        ).start()

    except Exception as e:
        logging.error(f"TTS synthesis failed: {e}", exc_info=True)
