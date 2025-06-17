"""
Voice package for Ratatoskr AI Assistant.

This package contains speech processing capabilities including text-to-speech
and speech-to-text functionality.

Modules:
- text_to_speech: TTS using Coqui TTS library
- speech_to_text: STT using Whisper and speech recognition
"""

from .text_to_speech import speak, speak_sync
from .speech_to_text import (
    listen_for_command,
    listen_for_command_continuous,
    get_stt_info,
    change_model as change_stt_model,
    test_microphone
)

__all__ = [
    'speak',
    'speak_sync',
    'listen_for_command',
    'listen_for_command_continuous',
    'get_stt_info',
    'change_stt_model',
    'test_microphone'
]
__version__ = '1.0.0' 