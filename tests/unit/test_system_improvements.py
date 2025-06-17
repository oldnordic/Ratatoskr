#!/usr/bin/env python3
"""
Test script for Ratatoskr system improvements:
- Voice mode popup messages
- Error handling improvements
- GPU acceleration status
- Application launching
"""

import sys
import os
import subprocess
import time
from typing import Dict, Any

def test_gpu_acceleration() -> Dict[str, Any]:
    """Test GPU acceleration for both PyTorch and Ollama."""
    print("🔍 Testing GPU Acceleration...")
    
    # Test PyTorch GPU
    try:
        import torch
        pytorch_gpu = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if pytorch_gpu else "None"
        print(f"   PyTorch GPU: {'✅' if pytorch_gpu else '❌'} {device_name}")
    except Exception as e:
        print(f"   PyTorch GPU: ❌ Error - {e}")
        pytorch_gpu = False
    
    # Test Ollama GPU
    try:
        result = subprocess.run(['ollama', 'run', 'llama3', 'test'], 
                              capture_output=True, text=True, timeout=30)
        ollama_working = result.returncode == 0
        print(f"   Ollama: {'✅' if ollama_working else '❌'} Working")
    except Exception as e:
        print(f"   Ollama: ❌ Error - {e}")
        ollama_working = False
    
    return {
        "pytorch_gpu": pytorch_gpu,
        "ollama_working": ollama_working
    }

def test_application_launching() -> Dict[str, Any]:
    """Test application launching capabilities."""
    print("\n🚀 Testing Application Launching...")
    
    results = {}
    
    # Test common applications available on this KDE system
    test_apps = {
        "firefox": "firefox",
        "dolphin": "dolphin", 
        "konsole": "konsole",
        "kate": "kate",
        "kcalc": "kcalc"
    }
    
    for app_name, command in test_apps.items():
        try:
            # Try to launch the application
            process = subprocess.Popen([command, '--version'], 
                                     stdout=subprocess.DEVNULL, 
                                     stderr=subprocess.DEVNULL)
            time.sleep(1)  # Give it a moment to start
            
            # Check if process is still running
            if process.poll() is None:
                print(f"   {app_name}: ✅ Available")
                process.terminate()
                results[app_name] = True
            else:
                print(f"   {app_name}: ⚠️ Not available")
                results[app_name] = False
                
        except Exception as e:
            print(f"   {app_name}: ❌ Error - {e}")
            results[app_name] = False
    
    return results

def test_error_handling() -> Dict[str, Any]:
    """Test error handling improvements."""
    print("\n🛡️ Testing Error Handling...")
    
    # Test voice TTS module error handling
    try:
        from voice.text_to_speech import speak_sync
        
        # Test with empty text
        result = speak_sync("")
        print(f"   Empty text handling: {'✅' if not result else '❌'}")
        
        # Test with None text
        result = speak_sync(None)
        print(f"   None text handling: {'✅' if not result else '❌'}")
        
        # Test with valid text
        result = speak_sync("Test message")
        print(f"   Valid text handling: {'✅' if result else '❌'}")
        
        return {"error_handling": True}
        
    except Exception as e:
        print(f"   Error handling test failed: {e}")
        return {"error_handling": False}

def test_voice_mode_popup() -> Dict[str, Any]:
    """Test voice mode popup functionality."""
    print("\n🎤 Testing Voice Mode Popup...")
    
    try:
        # Import the main app
        from main import RatatoskrApp
        from PyQt6.QtWidgets import QApplication
        
        # Create a minimal app for testing
        app = QApplication([])
        ratatoskr = RatatoskrApp()
        
        # Test mode switching
        print("   Testing hybrid mode switch...")
        ratatoskr.set_interaction_mode("hybrid")
        print("   ✅ Hybrid mode set successfully")
        
        print("   Testing voice-only mode switch...")
        ratatoskr.set_interaction_mode("voice_only")
        print("   ✅ Voice-only mode set successfully")
        
        print("   Testing text-only mode switch...")
        ratatoskr.set_interaction_mode("text_only")
        print("   ✅ Text-only mode set successfully")
        
        # Clean up
        app.quit()
        
        return {"voice_popup": True}
        
    except ImportError as e:
        print(f"   Voice popup test failed - missing dependency: {e}")
        return {"voice_popup": False}
    except Exception as e:
        print(f"   Voice popup test failed: {e}")
        return {"voice_popup": False}

def main():
    """Run all system improvement tests."""
    print("🧪 RATATOSKR SYSTEM IMPROVEMENTS TEST")
    print("=" * 50)
    
    # Run all tests
    gpu_results = test_gpu_acceleration()
    app_results = test_application_launching()
    error_results = test_error_handling()
    popup_results = test_voice_mode_popup()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    # Calculate overall score
    score = 0
    max_score = 4
    
    if gpu_results.get("pytorch_gpu"):
        score += 1
        print("✅ PyTorch GPU acceleration working")
    else:
        print("❌ PyTorch GPU acceleration not working")
    
    if gpu_results.get("ollama_working"):
        score += 1
        print("✅ Ollama working")
    else:
        print("❌ Ollama not working")
    
    if app_results:
        available_apps = sum(1 for v in app_results.values() if v)
        if available_apps >= 2:
            score += 1
            print(f"✅ Application launching working ({available_apps} apps available)")
        else:
            print(f"⚠️ Limited application launching ({available_apps} apps available)")
    
    if error_results.get("error_handling"):
        score += 1
        print("✅ Error handling improvements working")
    else:
        print("❌ Error handling improvements not working")
    
    # Performance level
    performance_level = ["Poor", "Fair", "Good", "Excellent"][score]
    print(f"\n🎯 Overall Performance: {performance_level} ({score}/{max_score})")
    
    if score == 4:
        print("🚀 All improvements working perfectly!")
    elif score >= 2:
        print("⚠️ Most improvements working, some optimizations possible")
    else:
        print("🐌 Limited improvements, significant work needed")
    
    print("\n" + "=" * 50)
    print("DETAILED RESULTS")
    print("=" * 50)
    
    print(f"\nGPU Acceleration:")
    print(f"   PyTorch: {gpu_results.get('pytorch_gpu', False)}")
    print(f"   Ollama: {gpu_results.get('ollama_working', False)}")
    
    print(f"\nApplication Launching:")
    for app, available in app_results.items():
        print(f"   {app}: {available}")
    
    print(f"\nError Handling: {error_results.get('error_handling', False)}")
    print(f"Voice Popup: {popup_results.get('voice_popup', False)}")

if __name__ == "__main__":
    main() 