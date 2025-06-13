import logging
import torch
from TTS.api import TTS
import sounddevice as sd
import numpy as np
import threading

# --- Global TTS Engine State ---
tts_model = None
model_lock = threading.Lock()

# Use the GPU if available, otherwise fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_tts_model():
    """Lazily loads the Coqui TTS model on first use."""
    global tts_model
    with model_lock:
        if tts_model is None:
            logging.info("Loading Coqui TTS model on demand...")
            try:
                # This model will be downloaded automatically on the first run
                model_name = "tts_models/en/ljspeech/vits"
                tts_model = TTS(model_name).to(device)
                logging.info(f"Coqui TTS model '{model_name}' loaded successfully on {device}.")
            except Exception as e:
                logging.error(f"Failed to initialize Coqui TTS model: {e}")
    return tts_model

def speak(text_to_speak):
    """
    Adds text to a queue to be synthesized and spoken by the dedicated TTS thread.
    """
    if not text_to_speak.strip():
        return
    thread = threading.Thread(target=_speak_thread, args=(text_to_speak,))
    thread.daemon = True
    thread.start()

def _speak_thread(text):
    """Worker function to synthesize and play audio using Coqui TTS."""
    model = get_tts_model()
    if not model:
        logging.error("TTS model not available, cannot speak.")
        return

    try:
        logging.info(f"Coqui TTS synthesizing: '{text[:50]}...'")
        
        # --- CHANGE: Simplified the TTS call ---
        # This specific model does not require speaker or language parameters.
        # Providing them was causing the 'NoneType' error.
        wav = model.tts(text=text)
        
        # Play the audio
        sd.play(np.array(wav), samplerate=model.synthesizer.output_sample_rate)
        sd.wait()
        logging.info("Finished TTS playback.")
    except Exception as e:
        logging.error(f"Error during Coqui TTS synthesis or playback: {e}")