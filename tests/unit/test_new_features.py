#!/usr/bin/env python3
"""
Test script for new Ratatoskr features:
- Configuration system
- Voice selection (male/female)
- Ollama server configuration
- Gmail integration
"""

import sys
import logging
from config import config_manager, VoiceConfig, OllamaConfig, GmailConfig

def test_configuration_system():
    """Test the new configuration system."""
    print("🔧 Testing Configuration System...")
    
    # Test voice configuration
    voice_config = config_manager.get_voice_config()
    print(f"  Voice Gender: {voice_config.gender}")
    print(f"  Voice Model: {voice_config.model}")
    print(f"  Voice Speed: {voice_config.speed}")
    print(f"  Voice Temperature: {voice_config.temperature}")
    
    # Test Ollama configuration
    ollama_config = config_manager.get_ollama_config()
    print(f"  Ollama Model: {ollama_config.name}")
    print(f"  Ollama Server: {ollama_config.server_url}")
    print(f"  Ollama Timeout: {ollama_config.timeout}")
    
    # Test Gmail configuration
    gmail_config = config_manager.get_gmail_config()
    print(f"  Gmail Enabled: {gmail_config.enabled}")
    print(f"  Gmail Email: {gmail_config.email}")
    print(f"  Calendar Alerts: {gmail_config.calendar_alerts}")
    print(f"  Alert Times: {gmail_config.alert_times}")
    print(f"  Daily Summary: {gmail_config.daily_summary}")
    print(f"  Summary Time: {gmail_config.summary_time}")
    
    print("✅ Configuration system test completed\n")

def test_voice_selection():
    """Test voice selection functionality."""
    print("🎤 Testing Voice Selection...")
    
    try:
        from voice.text_to_speech import get_available_models, get_voice_by_gender, test_voice_model
        
        # Test available models
        models = get_available_models()
        print(f"  Available models: {len(models)}")
        for model_name, config in models.items():
            print(f"    - {config['name']} ({config['gender']})")
        
        # Test gender-based voice selection
        male_voice = get_voice_by_gender("male")
        female_voice = get_voice_by_gender("female")
        print(f"  Male voice model: {male_voice}")
        print(f"  Female voice model: {female_voice}")
        
        # Test voice model (quick test)
        print("  Testing voice model (this may take a moment)...")
        success = test_voice_model("tts_models/en/ljspeech/fast_pitch", "Hello, this is a test.")
        print(f"  Voice test result: {'✅ Success' if success else '❌ Failed'}")
        
    except Exception as e:
        print(f"  ❌ Voice test failed: {e}")
    
    print("✅ Voice selection test completed\n")

def test_ollama_configuration():
    """Test Ollama configuration."""
    print("🤖 Testing Ollama Configuration...")
    
    try:
        import requests
        
        ollama_config = config_manager.get_ollama_config()
        print(f"  Server URL: {ollama_config.server_url}")
        print(f"  Model: {ollama_config.name}")
        
        # Test connection
        response = requests.get(f"{ollama_config.server_url}/api/tags", timeout=5)
        if response.status_code == 200:
            print("  ✅ Ollama server connection successful")
        else:
            print(f"  ⚠️  Ollama server responded with status: {response.status_code}")
            
    except Exception as e:
        print(f"  ❌ Ollama test failed: {e}")
    
    print("✅ Ollama configuration test completed\n")

def test_gmail_integration():
    """Test Gmail integration."""
    print("📧 Testing Gmail Integration...")
    
    try:
        from gmail_integration import GmailService, GMAIL_AVAILABLE
        
        if not GMAIL_AVAILABLE:
            print("  ⚠️  Gmail libraries not available")
            return
        
        gmail_config = config_manager.get_gmail_config()
        print(f"  Gmail enabled: {gmail_config.enabled}")
        
        if gmail_config.enabled and gmail_config.email:
            print(f"  Email configured: {gmail_config.email}")
            
            # Test Gmail service creation
            gmail_service = GmailService(config_manager)
            print("  ✅ Gmail service created successfully")
            
            # Note: Authentication requires credentials.json file
            print("  ℹ️  Authentication requires credentials.json file")
        else:
            print("  ℹ️  Gmail not configured - enable in settings")
            
    except Exception as e:
        print(f"  ❌ Gmail test failed: {e}")
    
    print("✅ Gmail integration test completed\n")

def test_settings_dialog():
    """Test settings dialog creation."""
    print("⚙️  Testing Settings Dialog...")
    
    try:
        from PyQt6.QtWidgets import QApplication
        from settings_dialog import SettingsDialog
        
        # Create minimal app for testing
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        # Test dialog creation
        dialog = SettingsDialog()
        print("  ✅ Settings dialog created successfully")
        
        # Test dialog properties
        print(f"  Dialog title: {dialog.windowTitle()}")
        print(f"  Dialog size: {dialog.size().width()}x{dialog.size().height()}")
        
    except Exception as e:
        print(f"  ❌ Settings dialog test failed: {e}")
    
    print("✅ Settings dialog test completed\n")

def main():
    """Run all tests."""
    print("🧪 Ratatoskr New Features Test Suite\n")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    test_configuration_system()
    test_voice_selection()
    test_ollama_configuration()
    test_gmail_integration()
    test_settings_dialog()
    
    print("=" * 50)
    print("🎉 All tests completed!")
    print("\nNext steps:")
    print("1. Run the main application: python main.py")
    print("2. Open Settings (Ctrl+,) to configure voice, Ollama, and Gmail")
    print("3. For Gmail integration, download credentials.json from Google Cloud Console")
    print("4. Test voice selection by changing gender in settings")

if __name__ == "__main__":
    main() 