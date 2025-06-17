#!/usr/bin/env python3
"""
Test script to verify male and female voice functionality.
"""

import time
from voice.text_to_speech import speak_sync, get_voice_by_gender, get_male_speaker
from config import VoiceConfig, config_manager

def test_voice_gender():
    """Test both male and female voices."""
    print("=" * 60)
    print("VOICE GENDER TEST")
    print("=" * 60)
    
    test_text = "Hello, this is a test of the voice gender selection. You should hear a clear difference between male and female voices."
    
    # Test male voice
    print("\n🎤 Testing MALE voice...")
    print(f"Model: {get_voice_by_gender('male')}")
    print(f"Male speaker: {get_male_speaker()}")
    
    config_manager.update_voice_config(VoiceConfig(
        gender='male',
        model='tts_models/en/vctk/vits',
        speed=1.0,
        temperature=0.6
    ))
    
    start_time = time.time()
    speak_sync(test_text)
    male_time = time.time() - start_time
    print(f"✅ Male voice completed in {male_time:.2f}s")
    
    time.sleep(1)  # Pause between voices
    
    # Test female voice
    print("\n🎤 Testing FEMALE voice...")
    print(f"Model: {get_voice_by_gender('female')}")
    
    config_manager.update_voice_config(VoiceConfig(
        gender='female',
        model='tts_models/en/ljspeech/fast_pitch',
        speed=1.0,
        temperature=0.6
    ))
    
    start_time = time.time()
    speak_sync(test_text)
    female_time = time.time() - start_time
    print(f"✅ Female voice completed in {female_time:.2f}s")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Male voice: {get_voice_by_gender('male')} (speaker: {get_male_speaker()})")
    print(f"Female voice: {get_voice_by_gender('female')}")
    print(f"Male voice time: {male_time:.2f}s")
    print(f"Female voice time: {female_time:.2f}s")
    print("\n✅ Both voices should sound distinctly different!")
    print("   - Male voice should be deeper and more masculine")
    print("   - Female voice should be higher pitched and feminine")

if __name__ == "__main__":
    test_voice_gender() 