
#!/usr/bin/env python3
"""
Detailed timing test for speech recognition to identify bottlenecks.
"""

import time
import logging
from voice.speech_to_text import listen_for_command, DEFAULT_TIMEOUT, DEFAULT_PHRASE_TIME_LIMIT

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

def test_detailed_timing():
    """Test detailed timing of each step in speech recognition."""
    print("=" * 60)
    print("DETAILED SPEECH RECOGNITION TIMING TEST")
    print("=" * 60)
    print(f"Settings: timeout={DEFAULT_TIMEOUT}s, phrase_limit={DEFAULT_PHRASE_TIME_LIMIT}s")
    print()
    
    print("Please speak a short phrase when prompted...")
    print("This will measure each step of the process.")
    print()
    
    # Measure total time
    total_start = time.time()
    
    # The listen_for_command function will log its internal timing
    text = listen_for_command()
    
    total_end = time.time()
    total_duration = total_end - total_start
    
    print(f"\n" + "=" * 60)
    print("TIMING RESULTS")
    print("=" * 60)
    print(f"Total duration: {total_duration:.2f} seconds")
    print(f"Transcribed text: '{text}'")
    
    # Analyze the timing
    if total_duration > DEFAULT_TIMEOUT + 2:  # Allow 2s for processing
        print(f"⚠️  Total time ({total_duration:.2f}s) exceeds timeout ({DEFAULT_TIMEOUT}s) + processing time")
        print("   This suggests the timeout might not be working properly")
    else:
        print(f"✅ Total time ({total_duration:.2f}s) is within expected range")
    
    return text, total_duration

if __name__ == "__main__":
    test_detailed_timing() 