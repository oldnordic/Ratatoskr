#!/usr/bin/env python3
"""
Test VITS model with male speakers for better quality/speed balance.
"""

from TTS.api import TTS
import time

def test_vits_male_speakers():
    """Test VITS model with different male speakers."""
    print("=" * 60)
    print("TESTING VITS MODEL WITH MALE SPEAKERS")
    print("=" * 60)
    
    test_text = "Hello, this is a test of the VITS model with male voice. It should sound natural and be reasonably fast."
    
    try:
        # Load VITS model
        print("Loading VITS model...")
        tts = TTS(model_name="tts_models/en/vctk/vits")
        
        # Get available speakers
        speakers = tts.speakers
        print(f"Available speakers: {len(speakers)}")
        
        # Find male speakers (usually start with 'p' followed by numbers)
        male_speakers = []
        for speaker in speakers:
            if speaker.startswith('p') and speaker[1:].isdigit():
                # Check if it's a male speaker (usually p225-p376 are male)
                speaker_num = int(speaker[1:])
                if 225 <= speaker_num <= 376:
                    male_speakers.append(speaker)
        
        print(f"Male speakers found: {len(male_speakers)}")
        
        # Test a few male speakers
        test_speakers = male_speakers[:3] if len(male_speakers) >= 3 else male_speakers[:1]
        
        results = {}
        for speaker in test_speakers:
            print(f"\nTesting speaker: {speaker}")
            start_time = time.time()
            
            try:
                wav = tts.tts(text=test_text, speaker=speaker)
                synthesis_time = time.time() - start_time
                results[speaker] = synthesis_time
                print(f"✅ {speaker}: {synthesis_time:.2f}s ({len(test_text) / synthesis_time:.1f} chars/sec)")
            except Exception as e:
                print(f"❌ {speaker}: Error - {e}")
        
        # Show best result
        if results:
            best_speaker = min(results.items(), key=lambda x: x[1])
            print(f"\n🏆 Best male speaker: {best_speaker[0]} ({best_speaker[1]:.2f}s)")
            return best_speaker[0], best_speaker[1]
        
    except Exception as e:
        print(f"Error testing VITS: {e}")
        return None, None

def compare_all_models():
    """Compare all tested models."""
    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)
    
    test_text = "Hello, this is a speed test."
    
    models_to_test = [
        ("Current", "tts_models/en/ljspeech/tacotron2-DDC", None),
        ("Fast Pitch", "tts_models/en/ljspeech/fast_pitch", None),
    ]
    
    # Add VITS with best male speaker if available
    best_speaker, _ = test_vits_male_speakers()
    if best_speaker:
        models_to_test.append(("VITS Male", "tts_models/en/vctk/vits", best_speaker))
    
    results = {}
    for name, model_name, speaker in models_to_test:
        print(f"\nTesting {name}...")
        try:
            start_time = time.time()
            tts = TTS(model_name=model_name)
            load_time = time.time() - start_time
            
            start_time = time.time()
            if speaker:
                wav = tts.tts(text=test_text, speaker=speaker)
            else:
                wav = tts.tts(text=test_text)
            synthesis_time = time.time() - start_time
            
            total_time = load_time + synthesis_time
            results[name] = {
                'load_time': load_time,
                'synthesis_time': synthesis_time,
                'total_time': total_time,
                'chars_per_sec': len(test_text) / synthesis_time
            }
            
            print(f"✅ {name}: {synthesis_time:.2f}s synthesis, {total_time:.2f}s total")
            
        except Exception as e:
            print(f"❌ {name}: Error - {e}")
    
    # Show recommendations
    if results:
        print("\n" + "=" * 60)
        print("RECOMMENDATIONS")
        print("=" * 60)
        
        fastest_synthesis = min(results.items(), key=lambda x: x[1]['synthesis_time'])
        fastest_total = min(results.items(), key=lambda x: x[1]['total_time'])
        
        print(f"🏆 Fastest synthesis: {fastest_synthesis[0]} ({fastest_synthesis[1]['synthesis_time']:.2f}s)")
        print(f"🏆 Fastest total time: {fastest_total[0]} ({fastest_total[1]['total_time']:.2f}s)")
        
        print("\n📊 Detailed Results:")
        for name, data in results.items():
            print(f"  {name}:")
            print(f"    Load time: {data['load_time']:.2f}s")
            print(f"    Synthesis time: {data['synthesis_time']:.2f}s")
            print(f"    Total time: {data['total_time']:.2f}s")
            print(f"    Speed: {data['chars_per_sec']:.1f} chars/sec")

if __name__ == "__main__":
    compare_all_models() 