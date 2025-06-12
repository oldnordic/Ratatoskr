import logging
import sounddevice as sd
import numpy as np
import threading
from piper.voice import PiperVoice

# --- Configuration ---
VOICE_MODEL_PATH = 'tts_models/en_US-ryan-high.onnx' 

# --- Initialization ---
voice = None
logging.info("Loading Piper TTS voice model...")
try:
    voice = PiperVoice.load(VOICE_MODEL_PATH)
    logging.info(f"Piper TTS model '{VOICE_MODEL_PATH}' loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load Piper TTS model from '{VOICE_MODEL_PATH}'. Error: {e}")

def speak(text_to_speak):
    """
    Synthesizes and speaks the given text using Piper TTS in a separate thread.
    """
    if not voice:
        logging.error("TTS voice not loaded, cannot speak.")
        return
    if not text_to_speak.strip():
        return
        
    thread = threading.Thread(target=_speak_thread, args=(text_to_speak,))
    thread.daemon = True
    thread.start()

def _speak_thread(text):
    """Worker function to synthesize and play audio."""
    try:
        audio_stream = voice.synthesize_stream_raw(text)
        samplerate = voice.config.sample_rate
        audio_data = b''.join(audio_stream)
            
        if audio_data:
             audio_np = np.frombuffer(audio_data, dtype=np.int16)
             sd.play(audio_np, samplerate=samplerate)
             
             # This line is critical. It blocks this background thread
             # until the sound is done, but does NOT block the main UI.
             sd.wait()

    except Exception as e:
        logging.error(f"Error during TTS synthesis or playback: {e}")