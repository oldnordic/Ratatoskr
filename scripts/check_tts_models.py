#!/usr/bin/env python3
"""
Check available Coqui TTS models and their characteristics for optimization.
"""

from TTS.api import TTS
import time

def list_available_models():
    """List available TTS models with their characteristics."""
    print("=" * 80)
    print("AVAILABLE COQUI TTS MODELS")
    print("=" * 80)
    
    try:
        tts = TTS()
        models = tts.list_models()
        
        print(f"Total models available: {len(models)}")
        print()
        
        # Filter for English models and show characteristics
        english_models = []
        for model in models:
            if 'en' in model.lower() or 'english' in model.lower():
                english_models.append(model)
        
        print("ENGLISH MODELS (Recommended for speed/quality balance):")
        print("-" * 60)
        
        for i, model in enumerate(english_models[:15], 1):
            print(f"{i:2d}. {model}")
            
            # Try to get model info
            try:
                model_tts = TTS(model_name=model)
                speakers = getattr(model_tts, 'speakers', None)
                if speakers:
                    print(f"     Speakers: {len(speakers)} available")
                print(f"     Type: {type(model_tts).__name__}")
            except Exception as e:
                print(f"     Error loading: {str(e)[:50]}...")
            print()
        
        print("=" * 80)
        print("OPTIMIZATION RECOMMENDATIONS")
        print("=" * 80)
        print("1. Look for models with 'fast' or 'small' in the name")
        print("2. Male voices often have better speed/quality balance")
        print("3. Models with fewer speakers are usually faster")
        print("4. Consider 'tts_models/en/ljspeech/tacotron2-DDC' (current)")
        print("5. Try 'tts_models/en/ljspeech/fast_pitch' for speed")
        print("6. Look for 'tts_models/en/vctk/vits' for quality/speed balance")
        
    except Exception as e:
        print(f"Error listing models: {e}")

def test_model_speed(model_name, test_text="Hello, this is a speed test."):
    """Test the speed of a specific model."""
    print(f"\nTesting model: {model_name}")
    print(f"Text: '{test_text}'")
    
    try:
        start_time = time.time()
        tts = TTS(model_name=model_name)
        load_time = time.time() - start_time
        print(f"Model load time: {load_time:.2f}s")
        
        start_time = time.time()
        wav = tts.tts(test_text)
        synthesis_time = time.time() - start_time
        print(f"Synthesis time: {synthesis_time:.2f}s")
        print(f"Total time: {load_time + synthesis_time:.2f}s")
        print(f"Characters per second: {len(test_text) / synthesis_time:.1f}")
        
        return synthesis_time
        
    except Exception as e:
        print(f"Error testing model: {e}")
        return None

def main():
    """Main function to check models and test speed."""
    print("COQUI TTS MODEL OPTIMIZATION CHECKER")
    print("This will help find faster models while maintaining quality.")
    
    # List available models
    list_available_models()
    
    # Test some specific models for speed
    print("\n" + "=" * 80)
    print("SPEED TESTING SPECIFIC MODELS")
    print("=" * 80)
    
    test_models = [
        "tts_models/en/ljspeech/tacotron2-DDC",  # Current
        "tts_models/en/ljspeech/fast_pitch",     # Fast alternative
        "tts_models/en/vctk/vits",               # Quality/speed balance
    ]
    
    results = {}
    for model in test_models:
        speed = test_model_speed(model)
        if speed:
            results[model] = speed
    
    # Show recommendations
    if results:
        print("\n" + "=" * 80)
        print("SPEED COMPARISON RESULTS")
        print("=" * 80)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1])
        for i, (model, speed) in enumerate(sorted_results, 1):
            print(f"{i}. {model}: {speed:.2f}s")
        
        fastest = sorted_results[0]
        print(f"\n🏆 Fastest model: {fastest[0]} ({fastest[1]:.2f}s)")
        print("Consider switching to this model for better speed!")

if __name__ == "__main__":
    main() 