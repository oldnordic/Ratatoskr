import logging
import sounddevice as sd
import wave
import threading
from piper.voice import PiperVoice # Correct import path

# --- Configuration ---
VOICE_MODEL_PATH = 'tts_models/en_US-ryan-high.onnx' # Path to the .onnx file

# --- Initialization ---
voice = None
logging.info("Loading Piper TTS voice model...")
try:
    # UPDATED: This is the new, correct way to load the voice model.
    voice = PiperVoice.load(VOICE_MODEL_PATH)
    logging.info("Piper TTS model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load Piper TTS model: {e}")

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
        # Create an in-memory audio buffer
        audio_stream = voice.synthesize_stream_raw(text)

        # Play the audio stream directly
        # Note: You might need to find the sample rate from the voice's config if not default
        # For now, we assume a common sample rate like 22050
        samplerate = voice.config.sample_rate
        
        # Collect audio data from the generator
        audio_data = b''
        for audio_bytes in audio_stream:
            audio_data += audio_bytes
            
        if audio_data:
             # sounddevice expects numpy array, let's convert it
             import numpy as np
             audio_np = np.frombuffer(audio_data, dtype=np.int16)
             sd.play(audio_np, samplerate=samplerate)
             sd.wait()

    except Exception as e:
        logging.error(f"Error during TTS synthesis or playback: {e}")