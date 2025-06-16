import logging
import threading

import numpy as np
import sounddevice as sd
import torch
from TTS.api import TTS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model once and move it to the chosen device
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
tts.to(DEVICE)


def speak(text: str) -> None:
    """Synthesize ``text`` to speech and play it asynchronously."""
    logging.info(f"Coqui TTS synthesizing: '{text}'")
    try:
        wav: np.ndarray = tts.tts(text)
        sample_rate: int = tts.synthesizer.output_sample_rate

        def _playback(data: np.ndarray, sr: int) -> None:
            sd.play(data, samplerate=sr)
            sd.wait()

        threading.Thread(target=_playback, args=(wav, sample_rate), daemon=True).start()
    except Exception as e:
        logging.error(f"TTS synthesis failed: {e}", exc_info=True)
