#!/usr/bin/env python3
"""
Comprehensive test script to verify speech-to-text improvements.
Tests both speed and callback functionality.
"""

import time
import logging
import threading
from queue import Queue
from voice.speech_to_text import listen_for_command, DEFAULT_TIMEOUT, DEFAULT_PHRASE_TIME_LIMIT

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

def test_speed_improvements():
    """Test the speed improvements in speech recognition."""
    print("=" * 60)
    print("TESTING SPEECH-TO-TEXT SPEED IMPROVEMENTS")
    print("=" * 60)
    print(f"Current settings:")
    print(f"  - Timeout: {DEFAULT_TIMEOUT}s (was 5s)")
    print(f"  - Phrase limit: {DEFAULT_PHRASE_TIME_LIMIT}s (was 15s)")
    print(f"  - Pause threshold: 0.8s (was 1.5s)")
    print(f"  - Ambient noise duration: 0.3s (was 0.5s)")
    print()
    
    print("Please speak a short phrase when prompted...")
    print("(This will test the new faster timeouts)")
    
    start_time = time.time()
    text = listen_for_command()
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"\nResults:")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Transcribed: '{text}'")
    
    if duration < 4.0:  # Should be faster than old 5s timeout
        print("✅ Speed improvement confirmed!")
    else:
        print("❌ Speed improvement not working as expected")
    
    return text, duration

def test_callback_simulation():
    """Simulate the callback mechanism to test reliability."""
    print("\n" + "=" * 60)
    print("TESTING CALLBACK MECHANISM")
    print("=" * 60)
    
    result_queue = Queue()
    
    def callback_simulation(text):
        """Simulate the callback that should be triggered."""
        print(f"  CALLBACK TRIGGERED with text: '{text}'")
        result_queue.put(text)
    
    def background_recognition():
        """Simulate background speech recognition."""
        print("  Starting background speech recognition...")
        text = listen_for_command()
        print(f"  Background recognition completed: '{text}'")
        # Simulate the Qt signal emission
        callback_simulation(text)
    
    print("Please speak another phrase when prompted...")
    print("(This will test the callback mechanism)")
    
    # Start background thread
    thread = threading.Thread(target=background_recognition, daemon=True)
    thread.start()
    
    # Wait for result with timeout
    try:
        result = result_queue.get(timeout=10)
        print(f"✅ Callback mechanism working! Received: '{result}'")
        return True
    except:
        print("❌ Callback mechanism failed - no result received")
        return False

def main():
    """Run comprehensive voice tests."""
    print("COMPREHENSIVE VOICE PROCESSING TEST")
    print("This test will verify both speed improvements and callback reliability.")
    print()
    
    # Test 1: Speed improvements
    text, duration = test_speed_improvements()
    
    # Test 2: Callback mechanism
    callback_working = test_callback_simulation()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Speed test: {'✅ PASSED' if duration < 4.0 else '❌ FAILED'}")
    print(f"Callback test: {'✅ PASSED' if callback_working else '❌ FAILED'}")
    
    if duration < 4.0 and callback_working:
        print("\n🎉 All tests passed! Voice processing improvements are working.")
    else:
        print("\n⚠️  Some tests failed. Check the issues above.")

if __name__ == "__main__":
    main() 