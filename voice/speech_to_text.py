import speech_recognition as sr
import whisper
import logging
import os
import wave
from tempfile import NamedTemporaryFile

# --- Initialization ---
# --- CHANGE: Upgraded the model from "tiny.en" to "small.en" for better accuracy ---
logging.info("Loading local Whisper STT model (small.en)...")
stt_model = whisper.load_model("small.en")
logging.info("Whisper STT model loaded.")


def listen_for_command():
    """
    Listens for a command from the microphone and returns the transcribed text
    using the local Whisper model.
    """
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1.5

    with sr.Microphone() as source:
        logging.info("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        logging.info("Listening for audio...")
        try:
            audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            logging.info("Audio captured, now transcribing...")
        except sr.WaitTimeoutError:
            logging.warning("Listening timed out while waiting for phrase to start")
            return ""

    # Create a temporary file to save the audio data
    temp_audio_file = NamedTemporaryFile(suffix=".wav", delete=False).name
    with wave.open(temp_audio_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio_data.sample_width)
        wf.setframerate(audio_data.sample_rate)
        wf.writeframes(audio_data.get_wav_data())

    try:
        # Transcribe the audio file with Whisper
        result = stt_model.transcribe(temp_audio_file, fp16=False)
        transcribed_text = result['text']
        logging.info(f"Whisper transcribed: '{transcribed_text}'")
        return transcribed_text
    except Exception as e:
        logging.error(f"Error during Whisper transcription: {e}")
        return ""
    finally:
        # Clean up the temporary file
        os.remove(temp_audio_file)