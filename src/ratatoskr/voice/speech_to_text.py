"""
Speech-to-text processing using Whisper and speech recognition.

This module provides high-quality speech recognition capabilities using
OpenAI's Whisper model combined with speech_recognition for microphone
input handling. It includes features like ambient noise adjustment,
timeout handling, and temporary file management.

Key Features:
- Local Whisper model for offline transcription
- Ambient noise calibration
- Configurable timeouts and phrase limits
- Temporary file cleanup
- Error handling and logging
"""

import logging
import os
import threading
import wave
from tempfile import NamedTemporaryFile
from typing import Optional

import speech_recognition as sr
import whisper

# Configuration constants
DEFAULT_MODEL_NAME = "tiny.en"  # much faster than 'small.en', still good quality
DEFAULT_PAUSE_THRESHOLD = 0.8  # seconds - reduced for faster response
DEFAULT_TIMEOUT = 3  # seconds - reduced for faster response
DEFAULT_PHRASE_TIME_LIMIT = 10  # seconds - reduced for faster response
DEFAULT_AMBIENT_NOISE_DURATION = 0.3  # seconds - reduced for faster startup
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH = 2  # 16-bit audio

# Global Whisper model instance
_stt_model: Optional[whisper.Whisper] = None
_model_lock = threading.Lock()


def _get_stt_model() -> whisper.Whisper:
    """
    Get or create the Whisper speech-to-text model with thread safety.
    
    Returns:
        whisper.Whisper: The initialized Whisper model instance
    """
    global _stt_model
    
    with _model_lock:
        if _stt_model is None:
            logging.info(f"Loading Whisper STT model '{DEFAULT_MODEL_NAME}'...")
            try:
                _stt_model = whisper.load_model(DEFAULT_MODEL_NAME)
                logging.info("Whisper STT model loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load Whisper model: {e}")
                raise
    
    return _stt_model


def listen_for_command(
    timeout: int = DEFAULT_TIMEOUT,
    phrase_time_limit: int = DEFAULT_PHRASE_TIME_LIMIT,
    pause_threshold: float = DEFAULT_PAUSE_THRESHOLD
) -> str:
    """
    Capture audio from the microphone and return the transcribed text.
    
    This function handles the complete pipeline from microphone input
    to text transcription, including noise calibration, audio capture,
    and Whisper processing.
    
    Args:
        timeout: Maximum time to wait for speech to start (seconds)
        phrase_time_limit: Maximum duration of speech to capture (seconds)
        pause_threshold: Silence duration to mark end of speech (seconds)
        
    Returns:
        str: Transcribed text, or empty string if no speech detected
    """
    try:
        # Get Whisper model
        model = _get_stt_model()
        
        # Initialize speech recognizer
        recognizer = sr.Recognizer()
        recognizer.pause_threshold = pause_threshold
        
        # Configure microphone input
        with sr.Microphone() as source:
            logging.info("Calibrating microphone for ambient noise...")
            
            # Calibrate to background noise
            recognizer.adjust_for_ambient_noise(
                source, 
                duration=int(DEFAULT_AMBIENT_NOISE_DURATION)
            )
            
            logging.info(f"Listening for audio (timeout: {timeout}s, phrase limit: {phrase_time_limit}s)...")
            
            try:
                # Capture audio from microphone
                audio_data = recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_time_limit
                )
                
            except sr.WaitTimeoutError:
                logging.info("No speech detected within timeout period")
                logging.info("listen_for_command result: <empty>")
                return ""
            except Exception as e:
                logging.error(f"Speech recognition error: {e}")
                logging.info("listen_for_command result: <empty>")
                return ""
        
        # Process audio with Whisper
        result = _process_audio_with_whisper(audio_data, model)
        logging.info(f"listen_for_command result: '{result}'")
        return result
        
    except Exception as e:
        logging.error(f"Error in speech recognition: {e}", exc_info=True)
        logging.info("listen_for_command result: <empty>")
        return ""


def _process_audio_with_whisper(audio_data: sr.AudioData, model: whisper.Whisper) -> str:
    """
    Process audio data with Whisper for transcription.
    
    Args:
        audio_data: Audio data from speech recognition
        model: Whisper model instance
        
    Returns:
        str: Transcribed text
    """
    temp_audio_file = None
    
    try:
        # Create temporary WAV file for Whisper
        temp_audio_file = NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio_path = temp_audio_file.name
        
        # Write audio data to WAV file
        with wave.open(temp_audio_path, "wb") as wf:
            wf.setnchannels(DEFAULT_CHANNELS)
            wf.setsampwidth(audio_data.sample_width)
            wf.setframerate(audio_data.sample_rate)
            wf.writeframes(audio_data.get_wav_data())
        
        logging.debug(f"Audio saved to temporary file: {temp_audio_path}")
        
        # Transcribe with Whisper
        result = model.transcribe(temp_audio_path, fp16=False)
        transcribed_text = result.get("text", "").strip()
        
        logging.info(f"Transcription completed: '{transcribed_text[:100]}...'")
        return transcribed_text
        
    except Exception as e:
        logging.error(f"Error during Whisper transcription: {e}")
        return ""
        
    finally:
        # Clean up temporary file
        if temp_audio_file and os.path.exists(temp_audio_file.name):
            try:
                os.remove(temp_audio_file.name)
                logging.debug("Temporary audio file cleaned up")
            except Exception as e:
                logging.warning(f"Failed to clean up temporary file: {e}")


def listen_for_command_continuous(
    callback: callable,
    stop_event: Optional[threading.Event] = None,
    timeout: int = DEFAULT_TIMEOUT
) -> None:
    """
    Continuously listen for speech and call callback with results.
    
    This function runs in a loop, continuously listening for speech
    and calling the provided callback function with transcribed text.
    
    Args:
        callback: Function to call with transcribed text
        stop_event: Optional threading.Event to stop listening
        timeout: Timeout for each listening session
    """
    if stop_event is None:
        stop_event = threading.Event()
    
    logging.info("Starting continuous speech recognition...")
    
    while not stop_event.is_set():
        try:
            text = listen_for_command(timeout=timeout)
            if text:
                callback(text)
        except Exception as e:
            logging.error(f"Error in continuous speech recognition: {e}")
            if stop_event.is_set():
                break
    
    logging.info("Continuous speech recognition stopped")


def get_available_microphones() -> list:
    """
    Get a list of available microphone devices.
    
    Returns:
        list: List of available microphone names
    """
    try:
        return sr.Microphone.list_microphone_names()
    except Exception as e:
        logging.error(f"Failed to get microphone list: {e}")
        return []


def get_stt_info() -> dict:
    """
    Get information about the current STT setup.
    
    Returns:
        dict: STT configuration and status information
    """
    try:
        model = _get_stt_model()
        return {
            "model_name": DEFAULT_MODEL_NAME,
            "model_loaded": _stt_model is not None,
            "pause_threshold": DEFAULT_PAUSE_THRESHOLD,
            "timeout": DEFAULT_TIMEOUT,
            "phrase_time_limit": DEFAULT_PHRASE_TIME_LIMIT,
            "available_microphones": get_available_microphones()
        }
    except Exception as e:
        logging.error(f"Failed to get STT info: {e}")
        return {"error": str(e)}


def change_model(model_name: str) -> bool:
    """
    Change the Whisper model to a different one.
    
    Args:
        model_name: Name of the new Whisper model
        
    Returns:
        bool: True if model change was successful, False otherwise
    """
    global _stt_model
    
    try:
        logging.info(f"Changing Whisper model to: {model_name}")
        
        with _model_lock:
            # Load new model
            new_model = whisper.load_model(model_name)
            
            # Replace old model
            _stt_model = new_model
            
        logging.info(f"Successfully changed Whisper model to: {model_name}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to change Whisper model: {e}")
        return False


def test_microphone() -> dict:
    """
    Test microphone functionality and return diagnostic information.
    
    Returns:
        dict: Microphone test results and diagnostics
    """
    try:
        # Test microphone access
        with sr.Microphone() as source:
            recognizer = sr.Recognizer()
            
            # Test ambient noise adjustment
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            # Test audio capture (short duration)
            audio_data = recognizer.listen(source, timeout=2, phrase_time_limit=3)
            
            return {
                "microphone_accessible": True,
                "audio_captured": True,
                "sample_rate": audio_data.sample_rate,
                "sample_width": audio_data.sample_width,
                "duration": len(audio_data.frame_data) / (audio_data.sample_rate * audio_data.sample_width)
            }
            
    except Exception as e:
        logging.error(f"Microphone test failed: {e}")
        return {
            "microphone_accessible": False,
            "error": str(e)
        }
