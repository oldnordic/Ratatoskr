import speech_recognition as sr
import whisper
import logging
import os
import wave
import threading
from tempfile import NamedTemporaryFile

# --- CHANGE: Defer model loading ---
stt_model = None
model_lock = threading.Lock()

def get_stt_model():
    """Lazily loads the Whisper model on first use."""
    global stt_model
    with model_lock:
        if stt_model is None:
            logging.info("Loading local Whisper STT model (small.en) on demand...")
            stt_model = whisper.load_model("small.en")
            logging.info("Whisper STT model loaded.")
    return stt_model

def listen_for_command():
    """
    Listens for a command from the microphone and returns the transcribed text.
    """
    model = get_stt_model() # Load the model on first call
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1.5

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        logging.info("Listening for audio...")
        try:
            audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=15)
        except sr.WaitTimeoutError:
            return ""

    temp_audio_file = NamedTemporaryFile(suffix=".wav", delete=False).name
    with wave.open(temp_audio_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio_data.sample_width)
        wf.setframerate(audio_data.sample_rate)
        wf.writeframes(audio_data.get_wav_data())

    try:
        result = model.transcribe(temp_audio_file, fp16=False)
        return result.get('text', '')
    except Exception as e:
        logging.error(f"Error during Whisper transcription: {e}")
        return ""
    finally:
        os.remove(temp_audio_file)