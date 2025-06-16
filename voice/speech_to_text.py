"""Microphone input and speech-to-text utilities using Whisper."""

import logging
import os
import threading
import wave
from tempfile import NamedTemporaryFile

import speech_recognition as sr
import whisper

# These globals hold the Whisper model instance and a lock to ensure that it is
# loaded only once even when multiple threads access ``get_stt_model`` at the
# same time.
stt_model = None
model_lock = threading.Lock()


def get_stt_model():
    """Lazily load and return the Whisper speech-to-text model."""
    global stt_model
    with model_lock:
        if stt_model is None:
            # Loading the model can be expensive, so we do it only once and
            # reuse the instance for subsequent calls.
            logging.info(
                "Loading local Whisper STT model (small.en) on demand..."
            )
            stt_model = whisper.load_model("small.en")
            logging.info("Whisper STT model loaded.")
    return stt_model


def listen_for_command() -> str:
    """Capture audio from the microphone and return the transcribed text."""
    model = get_stt_model()
    # ``speech_recognition`` handles microphone input and converts it to an
    # ``AudioData`` object which we then feed to Whisper.
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1.5

    with sr.Microphone() as source:
        # Calibrate to background noise then wait for speech.
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        logging.info("Listening for audio...")
        try:
            audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=15)
        except sr.WaitTimeoutError:
            # If nothing was heard before the timeout, just return an empty string.
            return ""

    # Write the captured audio to a temporary WAV file for Whisper to read.
    temp_audio_file = NamedTemporaryFile(suffix=".wav", delete=False).name
    with wave.open(temp_audio_file, "wb") as wf:
        # Preserve the audio parameters so Whisper can transcribe accurately.
        wf.setnchannels(1)
        wf.setsampwidth(audio_data.sample_width)
        wf.setframerate(audio_data.sample_rate)
        wf.writeframes(audio_data.get_wav_data())

    try:
        result = model.transcribe(temp_audio_file, fp16=False)
        # ``transcribe`` returns a dict with a ``text`` key.
        return result.get("text", "")
    except Exception as e:
        logging.error(f"Error during Whisper transcription: {e}")
        return ""
    finally:
        # Clean up the temporary file regardless of success or failure.
        os.remove(temp_audio_file)
