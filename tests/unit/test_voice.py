"""
Comprehensive tests for the voice package.

This module contains unit tests for speech processing functionality including
text-to-speech, speech-to-text, and voice management features.

Test Coverage:
- Text-to-speech functionality
- Speech-to-text functionality
- Model management
- Error handling
- Performance benchmarks
"""

import unittest
from unittest.mock import patch, MagicMock, Mock
import tempfile
import os
import wave
import numpy as np

# Import the modules to test
from voice.text_to_speech import speak, speak_sync, get_tts_info, change_model
from voice.speech_to_text import (
    listen_for_command,
    listen_for_command_continuous,
    get_stt_info,
    change_model as change_stt_model,
    test_microphone
)


class TestTextToSpeech(unittest.TestCase):
    """Test suite for text-to-speech functionality."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.test_text = "Hello, this is a test message."
        self.test_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        self.test_file.close()
    
    def tearDown(self):
        """Clean up test environment after each test."""
        if os.path.exists(self.test_file.name):
            os.unlink(self.test_file.name)
    
    @patch('voice.text_to_speech.TTS')
    def test_speak_success(self, mock_tts_class):
        """Test successful text-to-speech conversion."""
        # Mock TTS instance
        mock_tts = MagicMock()
        mock_tts.tts_to_file.return_value = None
        mock_tts_class.return_value = mock_tts
        
        result = speak(self.test_text, self.test_file.name)
        
        self.assertTrue(result)
        mock_tts.tts_to_file.assert_called_once()
        
        # Verify the call arguments
        call_args = mock_tts.tts_to_file.call_args
        self.assertEqual(call_args[0][0], self.test_text)
        self.assertEqual(call_args[0][1], self.test_file.name)
    
    @patch('voice.text_to_speech.TTS')
    def test_speak_with_custom_model(self, mock_tts_class):
        """Test text-to-speech with custom model."""
        custom_model = "tts_models/en/ljspeech/tacotron2-DDC"
        
        # Mock TTS instance
        mock_tts = MagicMock()
        mock_tts.tts_to_file.return_value = None
        mock_tts_class.return_value = mock_tts
        
        result = speak(self.test_text, self.test_file.name, model=custom_model)
        
        self.assertTrue(result)
        mock_tts_class.assert_called_with(model=custom_model)
    
    @patch('voice.text_to_speech.TTS')
    def test_speak_tts_error(self, mock_tts_class):
        """Test text-to-speech with TTS error."""
        # Mock TTS error
        mock_tts = MagicMock()
        mock_tts.tts_to_file.side_effect = Exception("TTS error")
        mock_tts_class.return_value = mock_tts
        
        result = speak(self.test_text, self.test_file.name)
        
        self.assertFalse(result)
    
    def test_speak_empty_text(self):
        """Test text-to-speech with empty text."""
        result = speak("", self.test_file.name)
        self.assertFalse(result)
    
    def test_speak_none_text(self):
        """Test text-to-speech with None text."""
        result = speak(None, self.test_file.name)
        self.assertFalse(result)
    
    def test_speak_invalid_file_path(self):
        """Test text-to-speech with invalid file path."""
        result = speak(self.test_text, "/invalid/path/file.wav")
        self.assertFalse(result)
    
    @patch('voice.text_to_speech.TTS')
    def test_speak_sync_success(self, mock_tts_class):
        """Test synchronous text-to-speech."""
        # Mock TTS instance
        mock_tts = MagicMock()
        mock_tts.tts_to_file.return_value = None
        mock_tts_class.return_value = mock_tts
        
        result = speak_sync(self.test_text, self.test_file.name)
        
        self.assertTrue(result)
        mock_tts.tts_to_file.assert_called_once()
    
    @patch('voice.text_to_speech.TTS')
    def test_speak_sync_with_chunking(self, mock_tts_class):
        """Test synchronous text-to-speech with chunking."""
        long_text = "This is a very long text that should be chunked into smaller pieces for processing. " * 10
        
        # Mock TTS instance
        mock_tts = MagicMock()
        mock_tts.tts_to_file.return_value = None
        mock_tts_class.return_value = mock_tts
        
        result = speak_sync(long_text, self.test_file.name)
        
        self.assertTrue(result)
        # Should be called multiple times for chunking
        self.assertGreater(mock_tts.tts_to_file.call_count, 1)
    
    def test_get_tts_info(self):
        """Test TTS information retrieval."""
        result = get_tts_info()
        
        self.assertIsInstance(result, dict)
        self.assertIn('current_model', result)
        self.assertIn('available_models', result)
        self.assertIn('status', result)
    
    def test_change_model_success(self):
        """Test successful model change."""
        new_model = "tts_models/en/ljspeech/tacotron2-DDC"
        
        with patch('voice.text_to_speech.TTS') as mock_tts_class:
            mock_tts = MagicMock()
            mock_tts_class.return_value = mock_tts
            
            result = change_model(new_model)
            
            self.assertTrue(result)
    
    def test_change_model_invalid(self):
        """Test model change with invalid model."""
        invalid_model = "invalid_model"
        
        result = change_model(invalid_model)
        
        self.assertFalse(result)


class TestSpeechToText(unittest.TestCase):
    """Test suite for speech-to-text functionality."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.test_audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        self.test_audio_file.close()
        
        # Create a simple test audio file
        self._create_test_audio_file()
    
    def tearDown(self):
        """Clean up test environment after each test."""
        if os.path.exists(self.test_audio_file.name):
            os.unlink(self.test_audio_file.name)
    
    def _create_test_audio_file(self):
        """Create a simple test audio file."""
        # Create a simple sine wave
        sample_rate = 16000
        duration = 1.0  # 1 second
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * frequency * t)
        
        # Convert to 16-bit PCM
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Write to WAV file
        with wave.open(self.test_audio_file.name, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
    
    @patch('voice.speech_to_text.whisper.load_model')
    def test_listen_for_command_success(self, mock_load_model):
        """Test successful speech-to-text conversion."""
        # Mock Whisper model
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            'text': 'Hello, this is a test command.'
        }
        mock_load_model.return_value = mock_model
        
        result = listen_for_command(self.test_audio_file.name)
        
        self.assertIsInstance(result, str)
        self.assertEqual(result, 'Hello, this is a test command.')
        mock_model.transcribe.assert_called_once()
    
    @patch('voice.speech_to_text.whisper.load_model')
    def test_listen_for_command_empty_result(self, mock_load_model):
        """Test speech-to-text with empty result."""
        # Mock Whisper model with empty result
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            'text': ''
        }
        mock_load_model.return_value = mock_model
        
        result = listen_for_command(self.test_audio_file.name)
        
        self.assertEqual(result, '')
    
    @patch('voice.speech_to_text.whisper.load_model')
    def test_listen_for_command_whisper_error(self, mock_load_model):
        """Test speech-to-text with Whisper error."""
        # Mock Whisper error
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = Exception("Whisper error")
        mock_load_model.return_value = mock_model
        
        result = listen_for_command(self.test_audio_file.name)
        
        self.assertEqual(result, '')
    
    def test_listen_for_command_invalid_file(self):
        """Test speech-to-text with invalid audio file."""
        result = listen_for_command("/invalid/path/audio.wav")
        
        self.assertEqual(result, '')
    
    def test_listen_for_command_nonexistent_file(self):
        """Test speech-to-text with nonexistent file."""
        result = listen_for_command("nonexistent_file.wav")
        
        self.assertEqual(result, '')
    
    @patch('voice.speech_to_text.sr.Recognizer')
    @patch('voice.speech_to_text.sr.Microphone')
    def test_listen_for_command_continuous_success(self, mock_microphone, mock_recognizer):
        """Test continuous speech recognition."""
        # Mock recognizer
        mock_rec = MagicMock()
        mock_rec.listen.return_value = MagicMock()
        mock_rec.recognize_google.return_value = "Test command"
        mock_recognizer.return_value = mock_rec
        
        # Mock microphone
        mock_mic = MagicMock()
        mock_microphone.return_value = mock_mic
        
        # Test callback
        callback_called = False
        callback_result = None
        
        def test_callback(text):
            nonlocal callback_called, callback_result
            callback_called = True
            callback_result = text
        
        # Mock threading
        with patch('voice.speech_to_text.threading.Thread') as mock_thread:
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance
            
            listen_for_command_continuous(test_callback)
            
            # Verify thread was started
            mock_thread_instance.start.assert_called_once()
    
    @patch('voice.speech_to_text.sr.Recognizer')
    @patch('voice.speech_to_text.sr.Microphone')
    def test_listen_for_command_continuous_with_stop(self, mock_microphone, mock_recognizer):
        """Test continuous speech recognition with stop event."""
        from threading import Event
        
        # Mock recognizer
        mock_rec = MagicMock()
        mock_rec.listen.return_value = MagicMock()
        mock_rec.recognize_google.return_value = "Test command"
        mock_recognizer.return_value = mock_rec
        
        # Mock microphone
        mock_mic = MagicMock()
        mock_microphone.return_value = mock_mic
        
        # Create stop event
        stop_event = Event()
        
        def test_callback(text):
            pass
        
        # Mock threading
        with patch('voice.speech_to_text.threading.Thread') as mock_thread:
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance
            
            listen_for_command_continuous(test_callback, stop_event=stop_event)
            
            # Verify thread was started
            mock_thread_instance.start.assert_called_once()
    
    def test_get_stt_info(self):
        """Test STT information retrieval."""
        result = get_stt_info()
        
        self.assertIsInstance(result, dict)
        self.assertIn('current_model', result)
        self.assertIn('available_models', result)
        self.assertIn('status', result)
    
    def test_change_stt_model_success(self):
        """Test successful STT model change."""
        new_model = "base.en"
        
        result = change_stt_model(new_model)
        
        self.assertTrue(result)
    
    def test_change_stt_model_invalid(self):
        """Test STT model change with invalid model."""
        invalid_model = "invalid_model"
        
        result = change_stt_model(invalid_model)
        
        self.assertFalse(result)
    
    @patch('voice.speech_to_text.sr.Microphone')
    def test_test_microphone_success(self, mock_microphone):
        """Test microphone testing functionality."""
        # Mock microphone
        mock_mic = MagicMock()
        mock_microphone.return_value = mock_mic
        
        result = test_microphone()
        
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)
        self.assertIn('devices', result)
        self.assertIn('default_device', result)


class TestVoiceIntegration(unittest.TestCase):
    """Integration tests for voice package."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.test_text = "Integration test message"
        self.test_audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        self.test_audio_file.close()
    
    def tearDown(self):
        """Clean up integration test environment."""
        if os.path.exists(self.test_audio_file.name):
            os.unlink(self.test_audio_file.name)
    
    @patch('voice.text_to_speech.TTS')
    @patch('voice.speech_to_text.whisper.load_model')
    def test_tts_to_stt_workflow(self, mock_load_model, mock_tts_class):
        """Test complete TTS to STT workflow."""
        # Mock TTS
        mock_tts = MagicMock()
        mock_tts.tts_to_file.return_value = None
        mock_tts_class.return_value = mock_tts
        
        # Mock STT
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            'text': self.test_text
        }
        mock_load_model.return_value = mock_model
        
        # Generate speech
        tts_result = speak(self.test_text, self.test_audio_file.name)
        
        # Convert back to text
        stt_result = listen_for_command(self.test_audio_file.name)
        
        # Verify workflow
        self.assertTrue(tts_result)
        self.assertEqual(stt_result, self.test_text)
    
    @patch('voice.text_to_speech.TTS')
    def test_voice_model_management(self, mock_tts_class):
        """Test voice model management workflow."""
        # Mock TTS
        mock_tts = MagicMock()
        mock_tts.tts_to_file.return_value = None
        mock_tts_class.return_value = mock_tts
        
        # Get initial info
        initial_info = get_tts_info()
        
        # Change model
        new_model = "tts_models/en/ljspeech/tacotron2-DDC"
        change_result = change_model(new_model)
        
        # Get updated info
        updated_info = get_tts_info()
        
        # Verify model change
        self.assertTrue(change_result)
        self.assertNotEqual(initial_info['current_model'], updated_info['current_model'])


class TestVoicePerformance(unittest.TestCase):
    """Performance tests for voice package."""
    
    def setUp(self):
        """Set up performance test environment."""
        self.test_text = "Performance test message " * 100  # Long text
        self.test_audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        self.test_audio_file.close()
    
    def tearDown(self):
        """Clean up performance test environment."""
        if os.path.exists(self.test_audio_file.name):
            os.unlink(self.test_audio_file.name)
    
    @patch('voice.text_to_speech.TTS')
    def test_tts_performance(self, mock_tts_class):
        """Test TTS performance with long text."""
        import time
        
        # Mock TTS
        mock_tts = MagicMock()
        mock_tts.tts_to_file.return_value = None
        mock_tts_class.return_value = mock_tts
        
        start_time = time.time()
        result = speak_sync(self.test_text, self.test_audio_file.name)
        end_time = time.time()
        
        execution_time = end_time - start_time
        self.assertLess(execution_time, 10.0)  # Should complete within 10 seconds
        self.assertTrue(result)
    
    @patch('voice.speech_to_text.whisper.load_model')
    def test_stt_performance(self, mock_load_model):
        """Test STT performance."""
        import time
        
        # Create a longer test audio file
        sample_rate = 16000
        duration = 5.0  # 5 seconds
        frequency = 440
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * frequency * t)
        audio_data = (audio_data * 32767).astype(np.int16)
        
        with wave.open(self.test_audio_file.name, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        # Mock Whisper
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            'text': 'Performance test result'
        }
        mock_load_model.return_value = mock_model
        
        start_time = time.time()
        result = listen_for_command(self.test_audio_file.name)
        end_time = time.time()
        
        execution_time = end_time - start_time
        self.assertLess(execution_time, 5.0)  # Should complete within 5 seconds
        self.assertEqual(result, 'Performance test result')


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2) 