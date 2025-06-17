#!/usr/bin/env python3
"""
Test script to demonstrate different TTS quality options and their speed differences.
"""

import time
import subprocess
import sys
from voice.text_to_speech import speak_sync

def test_current_tts():
    """Test the current Coqui TTS quality and speed."""
    print("=" * 60)
    print("TESTING CURRENT COQUI TTS (High Quality, Slower)")
    print("=" * 60)
    
    test_text = "Hello, this is a test of the current text-to-speech system. It should sound natural but may take a few seconds to synthesize."
    
    start_time = time.time()
    print(f"Text: '{test_text}'")
    print("Starting synthesis...")
    
    try:
        speak_sync(test_text)
        end_time = time.time()
        duration = end_time - start_time
        print(f"✅ Coqui TTS completed in {duration:.2f} seconds")
        print(f"   Quality: High (Natural sounding)")
        print(f"   Speed: Slow ({duration:.2f}s for {len(test_text)} characters)")
    except Exception as e:
        print(f"❌ Coqui TTS failed: {e}")

def test_system_tts_options():
    """Test system TTS options if available."""
    print("\n" + "=" * 60)
    print("SYSTEM TTS OPTIONS (Fast, Lower Quality)")
    print("=" * 60)
    
    test_text = "Hello, this is a test of system text-to-speech. It should be very fast but may sound robotic."
    
    # Test espeak if available
    try:
        print("Testing espeak (if available)...")
        start_time = time.time()
        result = subprocess.run(['espeak', test_text], 
                              capture_output=True, text=True, timeout=10)
        end_time = time.time()
        if result.returncode == 0:
            duration = end_time - start_time
            print(f"✅ espeak completed in {duration:.2f} seconds")
            print(f"   Quality: Low (Robotic)")
            print(f"   Speed: Very Fast ({duration:.2f}s for {len(test_text)} characters)")
        else:
            print("❌ espeak not available or failed")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ espeak not available")
    
    # Test festival if available
    try:
        print("\nTesting festival (if available)...")
        start_time = time.time()
        result = subprocess.run(['festival', '--tts'], 
                              input=test_text, capture_output=True, text=True, timeout=10)
        end_time = time.time()
        if result.returncode == 0:
            duration = end_time - start_time
            print(f"✅ festival completed in {duration:.2f} seconds")
            print(f"   Quality: Medium (Less robotic than espeak)")
            print(f"   Speed: Fast ({duration:.2f}s for {len(test_text)} characters)")
        else:
            print("❌ festival not available or failed")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ festival not available")

def show_quality_comparison():
    """Show a comparison of different TTS options."""
    print("\n" + "=" * 60)
    print("TTS QUALITY COMPARISON")
    print("=" * 60)
    print("1. Coqui TTS (Current)")
    print("   ✅ Quality: High - Natural sounding voice")
    print("   ❌ Speed: Slow - 3-5 seconds per sentence")
    print("   💾 Size: Large - ~500MB+ models")
    print()
    print("2. Festival TTS")
    print("   ✅ Quality: Medium - Less robotic")
    print("   ✅ Speed: Fast - ~0.5-1 second per sentence")
    print("   💾 Size: Medium - ~50MB")
    print()
    print("3. espeak TTS")
    print("   ✅ Quality: Low - Robotic but clear")
    print("   ✅ Speed: Very Fast - ~0.1-0.3 seconds per sentence")
    print("   💾 Size: Small - ~10MB")
    print()
    print("4. System TTS (Linux)")
    print("   ✅ Quality: Variable - Depends on system")
    print("   ✅ Speed: Very Fast - Instant")
    print("   💾 Size: None - Uses system libraries")

def main():
    """Run TTS quality tests."""
    print("TTS QUALITY AND SPEED TEST")
    print("This will test different text-to-speech options and their trade-offs.")
    print()
    
    # Test current TTS
    test_current_tts()
    
    # Test system options
    test_system_tts_options()
    
    # Show comparison
    show_quality_comparison()
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("For your use case, I recommend:")
    print()
    print("1. Keep Coqui TTS for longer responses (quality matters)")
    print("2. Add Festival/espeak for quick acknowledgments")
    print("3. Use a hybrid approach: fast TTS for short responses, quality TTS for long ones")
    print()
    print("Would you like me to implement a hybrid TTS system?")

if __name__ == "__main__":
    main() 