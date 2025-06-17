#!/usr/bin/env python3
"""
GPU Detection and Optimization Check for Ratatoskr.

This script checks GPU availability and usage for both PyTorch and Ollama,
and provides recommendations for optimization.
"""

import subprocess
import sys
import logging
from typing import Dict, Any, Optional

def check_pytorch_gpu() -> Dict[str, Any]:
    """Check PyTorch GPU availability and usage."""
    try:
        import torch
        
        info = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "memory_allocated": None,
            "memory_reserved": None,
            "memory_total": None
        }
        
        if torch.cuda.is_available():
            info["memory_allocated"] = torch.cuda.memory_allocated(0) / 1024**3  # GB
            info["memory_reserved"] = torch.cuda.memory_reserved(0) / 1024**3  # GB
            info["memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        return info
        
    except ImportError:
        return {"available": False, "error": "PyTorch not installed"}
    except Exception as e:
        return {"available": False, "error": str(e)}

def check_ollama_gpu() -> Dict[str, Any]:
    """Check Ollama GPU usage and configuration."""
    try:
        # Check if Ollama is running
        result = subprocess.run(['pgrep', 'ollama'], capture_output=True, text=True)
        if result.returncode != 0:
            return {"running": False, "error": "Ollama not running"}
        
        # Get Ollama info
        result = subprocess.run(['ollama', 'show', '--json'], capture_output=True, text=True)
        if result.returncode == 0:
            import json
            info = json.loads(result.stdout)
            return {
                "running": True,
                "info": info,
                "gpu_enabled": "gpu" in info.get("parameters", {}).lower() if info else False
            }
        else:
            return {"running": True, "error": "Could not get Ollama info"}
            
    except Exception as e:
        return {"running": False, "error": str(e)}

def check_system_gpu() -> Dict[str, Any]:
    """Check system GPU information."""
    try:
        # Check for NVIDIA GPU
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpus = []
            for line in lines:
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 5:
                        gpus.append({
                            "name": parts[0],
                            "memory_total": int(parts[1]),
                            "memory_used": int(parts[2]),
                            "memory_free": int(parts[3]),
                            "utilization": int(parts[4])
                        })
            return {"type": "nvidia", "gpus": gpus}
        
        # Check for AMD GPU
        result = subprocess.run(['rocm-smi', '--showproductname', '--showmeminfo', 'all'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return {"type": "amd", "info": result.stdout}
        
        # Check for Intel GPU
        result = subprocess.run(['lspci', '|', 'grep', 'VGA'], capture_output=True, text=True)
        if result.returncode == 0 and 'intel' in result.stdout.lower():
            return {"type": "intel", "info": result.stdout}
        
        return {"type": "unknown", "error": "No GPU detected"}
        
    except Exception as e:
        return {"type": "error", "error": str(e)}

def get_optimization_recommendations(pytorch_info: Dict[str, Any], ollama_info: Dict[str, Any], system_info: Dict[str, Any]) -> list:
    """Get optimization recommendations based on current setup."""
    recommendations = []
    
    # PyTorch recommendations
    if not pytorch_info.get("available"):
        recommendations.append("❌ PyTorch GPU acceleration not available")
        if "error" in pytorch_info:
            recommendations.append(f"   - Error: {pytorch_info['error']}")
        recommendations.append("   - Install PyTorch with CUDA support for better performance")
    else:
        recommendations.append("✅ PyTorch GPU acceleration available")
        if pytorch_info.get("device_name"):
            recommendations.append(f"   - GPU: {pytorch_info['device_name']}")
        if pytorch_info.get("memory_total"):
            recommendations.append(f"   - GPU Memory: {pytorch_info['memory_total']:.1f} GB")
    
    # Ollama recommendations
    if not ollama_info.get("running"):
        recommendations.append("❌ Ollama not running")
        recommendations.append("   - Start Ollama service for AI model inference")
    else:
        recommendations.append("✅ Ollama is running")
        if ollama_info.get("gpu_enabled"):
            recommendations.append("✅ Ollama GPU acceleration enabled")
        else:
            recommendations.append("⚠️ Ollama GPU acceleration not enabled")
            recommendations.append("   - Use 'ollama run llama2' with GPU models")
            recommendations.append("   - Check Ollama documentation for GPU setup")
    
    # System recommendations
    if system_info.get("type") == "nvidia":
        recommendations.append("✅ NVIDIA GPU detected")
        gpus = system_info.get("gpus", [])
        for i, gpu in enumerate(gpus):
            recommendations.append(f"   - GPU {i}: {gpu['name']}")
            recommendations.append(f"     Memory: {gpu['memory_used']}/{gpu['memory_total']} MB ({gpu['utilization']}% used)")
    elif system_info.get("type") == "amd":
        recommendations.append("✅ AMD GPU detected")
        recommendations.append("   - Consider using ROCm for PyTorch acceleration")
    elif system_info.get("type") == "intel":
        recommendations.append("⚠️ Intel GPU detected")
        recommendations.append("   - Limited GPU acceleration support")
    else:
        recommendations.append("❌ No dedicated GPU detected")
        recommendations.append("   - CPU-only mode will be slower")
    
    # Performance recommendations
    if pytorch_info.get("available") and ollama_info.get("gpu_enabled"):
        recommendations.append("🚀 Optimal setup detected!")
        recommendations.append("   - Both PyTorch and Ollama using GPU")
        recommendations.append("   - Maximum performance expected")
    elif pytorch_info.get("available") and not ollama_info.get("gpu_enabled"):
        recommendations.append("⚠️ Partial GPU acceleration")
        recommendations.append("   - PyTorch using GPU, Ollama using CPU")
        recommendations.append("   - Enable Ollama GPU for better AI performance")
    elif not pytorch_info.get("available") and ollama_info.get("gpu_enabled"):
        recommendations.append("⚠️ Partial GPU acceleration")
        recommendations.append("   - Ollama using GPU, PyTorch using CPU")
        recommendations.append("   - Install PyTorch GPU for better TTS performance")
    else:
        recommendations.append("🐌 CPU-only mode")
        recommendations.append("   - Consider GPU setup for better performance")
    
    return recommendations

def main():
    """Main function to run all GPU checks."""
    print("🔍 RATATOSKR GPU DETECTION & OPTIMIZATION")
    print("=" * 50)
    
    # Run all checks
    print("\n📊 Checking PyTorch GPU...")
    pytorch_info = check_pytorch_gpu()
    
    print("📊 Checking Ollama GPU...")
    ollama_info = check_ollama_gpu()
    
    print("📊 Checking System GPU...")
    system_info = check_system_gpu()
    
    # Display results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    
    print(f"\n🐍 PyTorch GPU:")
    if pytorch_info.get("available"):
        print(f"   ✅ Available: {pytorch_info['device_name']}")
        print(f"   📊 Memory: {pytorch_info['memory_total']:.1f} GB total")
        if pytorch_info['memory_allocated']:
            print(f"   💾 Used: {pytorch_info['memory_allocated']:.1f} GB")
    else:
        print(f"   ❌ Not available")
        if "error" in pytorch_info:
            print(f"   🔍 Error: {pytorch_info['error']}")
    
    print(f"\n🤖 Ollama GPU:")
    if ollama_info.get("running"):
        print(f"   ✅ Running")
        if ollama_info.get("gpu_enabled"):
            print(f"   🚀 GPU acceleration enabled")
        else:
            print(f"   ⚠️ GPU acceleration not enabled")
    else:
        print(f"   ❌ Not running")
        if "error" in ollama_info:
            print(f"   🔍 Error: {ollama_info['error']}")
    
    print(f"\n💻 System GPU:")
    if system_info.get("type") == "nvidia":
        print(f"   ✅ NVIDIA GPU detected")
        for i, gpu in enumerate(system_info.get("gpus", [])):
            print(f"   📊 GPU {i}: {gpu['name']}")
            print(f"   💾 Memory: {gpu['memory_used']}/{gpu['memory_total']} MB")
            print(f"   📈 Utilization: {gpu['utilization']}%")
    elif system_info.get("type") == "amd":
        print(f"   ✅ AMD GPU detected")
    elif system_info.get("type") == "intel":
        print(f"   ⚠️ Intel GPU detected")
    else:
        print(f"   ❌ No dedicated GPU")
    
    # Get recommendations
    print(f"\n" + "=" * 50)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 50)
    
    recommendations = get_optimization_recommendations(pytorch_info, ollama_info, system_info)
    for rec in recommendations:
        print(rec)
    
    print(f"\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    # Performance score
    score = 0
    max_score = 3
    
    if pytorch_info.get("available"):
        score += 1
    if ollama_info.get("gpu_enabled"):
        score += 1
    if system_info.get("type") in ["nvidia", "amd"]:
        score += 1
    
    performance_level = ["Poor", "Fair", "Good", "Excellent"][score]
    
    print(f"🎯 Performance Level: {performance_level} ({score}/{max_score})")
    
    if score == 3:
        print("🚀 Your system is optimally configured for maximum performance!")
    elif score == 2:
        print("⚠️ Good performance, but some optimizations possible")
    elif score == 1:
        print("🐌 Limited performance, consider GPU setup")
    else:
        print("❌ CPU-only mode, significant performance improvements possible with GPU")

if __name__ == "__main__":
    main()
