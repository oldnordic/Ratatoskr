"""
Text-to-Speech module for Ratatoskr AI Assistant.

This module provides text-to-speech functionality using Coqui TTS with:
- Multiple voice models (Fast Pitch, Tacotron2, VITS)
- Male and female voice options
- Streaming playback for responsiveness
- GPU acceleration support
- Configurable speed and temperature
"""

import logging
import threading
import time
from typing import Optional, Dict, Any
import numpy as np
import sounddevice as sd
import soundfile as sf
from io import BytesIO

try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logging.error("TTS not available. Install TTS: pip install TTS")

# Voice model configurations
VOICE_MODELS = {
    "tts_models/en/ljspeech/fast_pitch": {
        "name": "Fast Pitch (Female)",
        "gender": "female",
        "speed": 1.2,
        "temperature": 0.5,
        "description": "Fast, clear female voice"
    },
    "tts_models/en/ljspeech/tacotron2-DDC": {
        "name": "Tacotron2 (Female)", 
        "gender": "female",
        "speed": 1.0,
        "temperature": 0.7,
        "description": "High quality female voice"
    },
    "tts_models/en/vctk/vits": {
        "name": "VITS (Multiple Speakers)",
        "gender": "both",
        "speed": 1.0,
        "temperature": 0.6,
        "description": "High quality with multiple speaker options"
    }
}

# Male speaker selection for VITS model
MALE_SPEAKERS = ["p225", "p226", "p227", "p228", "p229", "p230", "p231", "p232", "p233", "p234"]

# Global TTS instance
_tts_instance = None
_current_model = None

def get_tts_instance(model_name: str = None) -> Optional[TTS]:
    """
    Get or create TTS instance with specified model.
    
    Args:
        model_name: Name of the TTS model to use
        
    Returns:
        TTS instance or None if not available
    """
    global _tts_instance, _current_model
    
    if not TTS_AVAILABLE:
        logging.error("TTS library not available")
        return None
    
    # If no model specified, use default from config
    if model_name is None:
        from config import config_manager
        voice_config = config_manager.get_voice_config()
        model_name = get_voice_by_gender(voice_config.gender)
    
    # Return existing instance if same model
    if _tts_instance and _current_model == model_name:
        return _tts_instance
    
    try:
        logging.info(f"Initializing TTS with model: {model_name}")
        _tts_instance = TTS(model_name)
        _current_model = model_name
        logging.info(f"TTS initialized successfully with model: {model_name}")
        return _tts_instance
    except Exception as e:
        logging.error(f"Failed to initialize TTS with model {model_name}: {e}")
        return None

def speak_sync(text: str, model_name: str = None, speed: float = None, 
               temperature: float = None, speaker: str = None, model_path: str = None) -> bool:
    """
    Convert text to speech and play it synchronously (non-blocking).
    
    Args:
        text: Text to convert to speech
        model_name: TTS model to use (optional, uses gender-based selection)
        speed: Speech speed multiplier (optional, uses config default)
        temperature: Voice temperature (optional, uses config default)
        speaker: Speaker name for multi-speaker models (optional)
        model_path: Path to custom model (optional)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Handle None and empty text
        if text is None:
            logging.warning("None text provided to speak_sync")
            return False
                
        if not text or not text.strip():
            logging.warning("Empty text provided to speak_sync")
            return False
        
        # Get default values from config if not provided
        if speed is None or temperature is None or model_name is None:
            from config import config_manager
            voice_config = config_manager.get_voice_config()
            
            if speed is None:
                speed = voice_config.speed
            if temperature is None:
                temperature = voice_config.temperature
            if model_name is None and model_path is None:
                model_name = get_voice_by_gender(voice_config.gender)
        
        # Get TTS instance
        if model_path:
            tts = get_tts_instance_from_path(model_path)
        else:
            tts = get_tts_instance(model_name)
        
        if tts is None:
            logging.error("Failed to initialize TTS")
            return False
        
        # Log model information safely
        model_info = model_name or model_path or "default"
        logging.info(f"Speaking text with model: {model_info}, speed: {speed}, temperature: {temperature}")
        
        # Generate audio based on model type
        if model_name == "tts_models/en/ljspeech/fast_pitch" or ("fast_pitch" in str(model_path or "") and model_path is None):
            audio = tts.tts(text=text, speed=speed, temperature=temperature)
        elif model_name == "tts_models/en/vctk/vits" or ("vctk" in str(model_path or "") and model_path is None):
            # For VITS model without specific speaker, check gender
            from config import config_manager
            voice_config = config_manager.get_voice_config()
            if voice_config.gender == "male":
                # Use male speaker for male gender
                male_speaker = get_male_speaker()
                logging.info(f"Using male speaker: {male_speaker}")
                audio = tts.tts(text=text, speaker=male_speaker, speed=speed, temperature=temperature)
            else:
                # Use default speaker for female
                audio = tts.tts(text=text, speed=speed, temperature=temperature)
        else:
            # Use default speaker
            audio = tts.tts(text=text, speed=speed, temperature=temperature)
        
        # Play audio without blocking
        if audio is not None and len(audio) > 0:
            logging.info(f"Playing audio of length: {len(audio)} samples")
            sd.play(audio, tts.synthesizer.output_sample_rate)
            # Don't call sd.wait() - let it play in background
            logging.info("Audio playback started (non-blocking)")
            return True
        else:
            logging.error("Generated audio is empty or None")
            return False
            
    except Exception as e:
        logging.error(f"Error in speak_sync: {e}", exc_info=True)
        return False

def speak(text: str, model_name: str = None, speed: float = None, 
          temperature: float = None, speaker: str = None) -> None:
    """
    Asynchronously convert text to speech and play it.
    
    Args:
        text: Text to convert to speech
        model_name: TTS model to use (optional, uses gender-based selection)
        speed: Speech speed multiplier (optional, uses config default)
        temperature: Voice temperature (optional, uses config default)
        speaker: Speaker name for multi-speaker models (optional)
    """
    if not text.strip():
        logging.warning("Empty text provided to speak")
        return
    
    def speak_thread():
        try:
            speak_sync(text, model_name, speed, temperature, speaker)
        except Exception as e:
            logging.error(f"Error in speak thread: {e}")
    
    # Run in background thread to avoid blocking
    thread = threading.Thread(target=speak_thread, daemon=True)
    thread.start()

def speak_chunked(text: str, chunk_size: int = 300, model_name: str = None, 
                  speed: float = None, temperature: float = None, speaker: str = None) -> None:
    """
    Convert text to speech in chunks for streaming playback.
    
    Args:
        text: Text to convert to speech
        chunk_size: Maximum characters per chunk
        model_name: TTS model to use (optional, uses gender-based selection)
        speed: Speech speed multiplier (optional, uses config default)
        temperature: Voice temperature (optional, uses config default)
        speaker: Speaker name for multi-speaker models (optional)
    """
    if not text.strip():
        logging.warning("Empty text provided to speak_chunked")
        return
    
    # Split text into sentences for natural breaks
    sentences = text.split('. ')
    chunks = []
    
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    logging.info(f"Split text into {len(chunks)} chunks for streaming playback")
    
    def speak_chunks():
        for i, chunk in enumerate(chunks):
            try:
                logging.info(f"Speaking chunk {i+1}/{len(chunks)}: '{chunk[:50]}...'")
                speak_sync(chunk, model_name, speed, temperature, speaker)
                # Small pause between chunks
                time.sleep(0.1)
            except Exception as e:
                logging.error(f"Error speaking chunk {i+1}: {e}")
    
    # Run in background thread
    thread = threading.Thread(target=speak_chunks, daemon=True)
    thread.start()

def get_available_models() -> Dict[str, Dict[str, Any]]:
    """
    Get available TTS models with their configurations.
    
    Returns:
        Dict of model configurations
    """
    return VOICE_MODELS.copy()

def get_voice_by_gender(gender: str) -> str:
    """
    Get appropriate voice model for specified gender.
    
    Args:
        gender: "male", "female", or "both"
        
    Returns:
        Model name for the gender
    """
    if gender == "male":
        return "tts_models/en/vctk/vits"  # Use VITS with male speakers
    elif gender == "female":
        return "tts_models/en/ljspeech/fast_pitch"  # Use Fast Pitch for female
    else:
        return "tts_models/en/vctk/vits"

def get_male_speaker() -> str:
    """
    Get a male speaker for VITS model.
    
    Returns:
        Male speaker name
    """
    return MALE_SPEAKERS[0]  # Use first male speaker by default

def get_vits_speakers() -> list:
    """
    Get available speakers for VITS model.
    
    Returns:
        List of speaker names
    """
    try:
        tts = get_tts_instance("tts_models/en/vctk/vits")
        if tts and hasattr(tts, 'speakers'):
            return tts.speakers
        return []
    except Exception as e:
        logging.error(f"Error getting VITS speakers: {e}")
        return []

def test_voice_model(model_name: str, test_text: str = "Hello, this is a test of the voice model.") -> bool:
    """
    Test a voice model with sample text (non-blocking).
    
    Args:
        model_name: Name of the model to test
        test_text: Text to use for testing
        
    Returns:
        bool: True if test successful, False otherwise
    """
    try:
        logging.info(f"Testing voice model: {model_name}")
        # Use the threaded speak function instead of speak_sync
        speak(test_text, model_name)
        return True
    except Exception as e:
        logging.error(f"Voice model test failed for {model_name}: {e}")
        return False

def get_tts_instance_from_path(model_path: str) -> Optional[TTS]:
    """
    Get TTS instance from a custom model path.
    
    Args:
        model_path: Path to the custom model
        
    Returns:
        TTS instance or None if not available
    """
    if not TTS_AVAILABLE:
        logging.error("TTS library not available")
        return None
    
    try:
        logging.info(f"Initializing TTS with custom model path: {model_path}")
        tts = TTS(model_path)
        logging.info(f"TTS initialized successfully with custom model: {model_path}")
        return tts
    except Exception as e:
        logging.error(f"Failed to initialize TTS with custom model path {model_path}: {e}")
        return None

