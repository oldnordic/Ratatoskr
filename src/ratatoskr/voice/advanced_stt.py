"""
Advanced speech recognition and dictation system for Ratatoskr.

This module provides continuous dictation capabilities with punctuation insertion,
formatting commands recognition, and real-time text processing.
"""

import logging
import threading
import time
import re
from typing import Optional, Callable, Dict, List, Any
from queue import Queue
import speech_recognition as sr
import whisper

from voice.speech_to_text import _get_stt_model

class DictationController:
    """
    Advanced dictation controller with continuous recognition.
    
    Provides real-time dictation with punctuation insertion,
    formatting commands, and voice feedback.
    """
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.is_listening = False
        self.is_paused = False
        self.text_callback: Optional[Callable[[str], None]] = None
        self.command_callback: Optional[Callable[[str], None]] = None
        self.error_callback: Optional[Callable[[str], None]] = None
        
        # Dictation settings
        self.auto_punctuation = True
        self.formatting_commands = True
        self.continuous_mode = True
        self.pause_threshold = 0.8
        self.phrase_time_limit = 10
        
        # Text processing
        self.current_text = ""
        self.sentence_buffer = []
        self.last_processed = ""
        
        # Formatting commands
        self.formatting_patterns = {
            r'\bnew paragraph\b': '\n\n',
            r'\bnew line\b': '\n',
            r'\bperiod\b': '.',
            r'\bcomma\b': ',',
            r'\bquestion mark\b': '?',
            r'\bexclamation mark\b': '!',
            r'\bcolon\b': ':',
            r'\bsemicolon\b': ';',
            r'\bquote\b': '"',
            r'\bend quote\b': '"',
            r'\bopen parenthesis\b': '(',
            r'\bclose parenthesis\b': ')',
            r'\bcapitalize\b': 'CAPITALIZE',
            r'\ball caps\b': 'ALL_CAPS',
            r'\bno caps\b': 'NO_CAPS',
            r'\bdelete word\b': 'DELETE_WORD',
            r'\bdelete sentence\b': 'DELETE_SENTENCE',
            r'\bundo\b': 'UNDO',
            r'\bclear\b': 'CLEAR'
        }
        
        # Whisper model for better accuracy
        self.whisper_model = None
        self.model_lock = threading.Lock()
        
        logging.info("DictationController initialized")
    
    def initialize_microphone(self) -> bool:
        """Initialize microphone for dictation."""
        try:
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            logging.info("Microphone initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Error initializing microphone: {e}")
            return False
    
    def get_whisper_model(self):
        """Get or create Whisper model with thread safety."""
        with self.model_lock:
            if self.whisper_model is None:
                try:
                    self.whisper_model = _get_stt_model()
                    logging.info("Whisper model loaded for dictation")
                except Exception as e:
                    logging.error(f"Error loading Whisper model: {e}")
                    return None
        return self.whisper_model
    
    def start_dictation(self, text_callback: Callable[[str], None],
                       command_callback: Optional[Callable[[str], None]] = None,
                       error_callback: Optional[Callable[[str], None]] = None) -> bool:
        """
        Start continuous dictation.
        
        Args:
            text_callback: Function called with transcribed text
            command_callback: Function called with formatting commands
            error_callback: Function called with error messages
            
        Returns:
            True if started successfully, False otherwise
        """
        if self.is_listening:
            logging.warning("Dictation already running")
            return False
        
        if not self.initialize_microphone():
            return False
        
        self.text_callback = text_callback
        self.command_callback = command_callback
        self.error_callback = error_callback
        self.is_listening = True
        self.is_paused = False
        self.current_text = ""
        self.sentence_buffer = []
        
        # Start dictation thread
        self.dictation_thread = threading.Thread(target=self._dictation_loop, daemon=True)
        self.dictation_thread.start()
        
        logging.info("Dictation started")
        return True
    
    def stop_dictation(self) -> bool:
        """Stop continuous dictation."""
        if not self.is_listening:
            logging.warning("Dictation not running")
            return False
        
        self.is_listening = False
        self.is_paused = False
        
        if hasattr(self, 'dictation_thread'):
            self.dictation_thread.join(timeout=2)
        
        logging.info("Dictation stopped")
        return True
    
    def pause_dictation(self):
        """Pause dictation temporarily."""
        self.is_paused = True
        logging.info("Dictation paused")
    
    def resume_dictation(self):
        """Resume paused dictation."""
        self.is_paused = False
        logging.info("Dictation resumed")
    
    def _dictation_loop(self):
        """Main dictation processing loop."""
        while self.is_listening:
            if self.is_paused:
                time.sleep(0.1)
                continue
            
            try:
                with self.microphone as source:
                    logging.debug("Listening for speech...")
                    
                    try:
                        audio = self.recognizer.listen(
                            source,
                            timeout=1,
                            phrase_time_limit=self.phrase_time_limit
                        )
                    except sr.WaitTimeoutError:
                        continue
                    except Exception as e:
                        logging.warning(f"Error capturing audio: {e}")
                        continue
                
                # Process audio with Whisper
                text = self._process_audio(audio)
                if text:
                    self._process_text(text)
                    
            except Exception as e:
                logging.error(f"Error in dictation loop: {e}")
                if self.error_callback:
                    self.error_callback(f"Dictation error: {e}")
                time.sleep(1)
    
    def _process_audio(self, audio: sr.AudioData) -> Optional[str]:
        """Process audio data with Whisper."""
        try:
            model = self.get_whisper_model()
            if not model:
                return None
            
            # Convert audio to temporary file for Whisper
            import tempfile
            import wave
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Save audio to WAV file
            with wave.open(temp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(audio.sample_width)
                wf.setframerate(audio.sample_rate)
                wf.writeframes(audio.get_wav_data())
            
            # Transcribe with Whisper
            result = model.transcribe(temp_path, fp16=False)
            transcribed_text = result.get("text", "").strip()
            
            # Clean up temporary file
            import os
            os.unlink(temp_path)
            
            return transcribed_text
            
        except Exception as e:
            logging.error(f"Error processing audio: {e}")
            return None
    
    def _process_text(self, text: str):
        """Process transcribed text with formatting and commands."""
        if not text:
            return
        
        logging.debug(f"Processing text: '{text}'")
        
        # Check for formatting commands
        if self.formatting_commands:
            command = self._extract_formatting_command(text)
            if command:
                if self.command_callback:
                    self.command_callback(command)
                return
        
        # Apply auto-punctuation
        if self.auto_punctuation:
            text = self._apply_auto_punctuation(text)
        
        # Add to sentence buffer
        self.sentence_buffer.append(text)
        
        # Process complete sentences
        self._process_sentences()
    
    def _extract_formatting_command(self, text: str) -> Optional[str]:
        """Extract formatting commands from text."""
        text_lower = text.lower()
        
        for pattern, command in self.formatting_patterns.items():
            if re.search(pattern, text_lower):
                logging.debug(f"Formatting command detected: {command}")
                return command
        
        return None
    
    def _apply_auto_punctuation(self, text: str) -> str:
        """Apply automatic punctuation to text."""
        # Basic sentence ending detection
        if text and not text[-1] in '.!?':
            # Check for sentence-ending patterns
            if any(word in text.lower() for word in ['thank you', 'please', 'okay', 'yes', 'no']):
                text += '.'
            elif text.lower().startswith(('what', 'where', 'when', 'why', 'how', 'who')):
                text += '?'
        
        return text
    
    def _process_sentences(self):
        """Process complete sentences from buffer."""
        if not self.sentence_buffer:
            return
        
        # Join sentences with proper spacing
        full_text = ' '.join(self.sentence_buffer)
        
        # Check if we have a complete sentence
        if self._is_complete_sentence(full_text):
            # Process the complete sentence
            processed_text = self._format_sentence(full_text)
            
            if self.text_callback:
                self.text_callback(processed_text)
            
            # Clear buffer
            self.sentence_buffer = []
            self.last_processed = processed_text
    
    def _is_complete_sentence(self, text: str) -> bool:
        """Check if text forms a complete sentence."""
        # Simple heuristic: ends with punctuation or is a short phrase
        if text.endswith(('.', '!', '?')):
            return True
        
        # Check for natural pause indicators
        pause_indicators = ['thank you', 'please', 'okay', 'yes', 'no', 'well', 'so']
        if any(indicator in text.lower() for indicator in pause_indicators):
            return True
        
        # If text is long enough, consider it complete
        if len(text.split()) >= 5:
            return True
        
        return False
    
    def _format_sentence(self, text: str) -> str:
        """Format sentence with proper capitalization and spacing."""
        # Capitalize first letter
        if text and text[0].isalpha():
            text = text[0].upper() + text[1:]
        
        # Ensure proper spacing
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def get_dictation_status(self) -> Dict[str, Any]:
        """Get current dictation status."""
        return {
            "is_listening": self.is_listening,
            "is_paused": self.is_paused,
            "auto_punctuation": self.auto_punctuation,
            "formatting_commands": self.formatting_commands,
            "continuous_mode": self.continuous_mode,
            "current_text": self.current_text,
            "sentence_buffer": self.sentence_buffer.copy(),
            "last_processed": self.last_processed
        }
    
    def update_settings(self, **kwargs):
        """Update dictation settings."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logging.info(f"Updated dictation setting: {key} = {value}")
    
    def clear_text(self):
        """Clear current text buffer."""
        self.current_text = ""
        self.sentence_buffer = []
        self.last_processed = ""
        logging.info("Text buffer cleared")

# Global dictation controller instance
dictation_controller = DictationController() 