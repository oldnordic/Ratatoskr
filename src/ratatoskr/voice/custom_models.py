"""
Custom Voice Model Management for Ratatoskr AI Assistant.

This module provides functionality for:
- Discovering available TTS models
- Downloading custom voice models
- Managing installed models
- Configuring voice model settings
"""

import os
import json
import logging
import subprocess
import webbrowser
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logging.error("TTS not available. Install TTS: pip install TTS")

# Custom models directory
CUSTOM_MODELS_DIR = Path("tts_models/custom")
CUSTOM_MODELS_CONFIG_FILE = CUSTOM_MODELS_DIR / "models.json"

@dataclass
class VoiceModel:
    """Voice model configuration."""
    name: str
    path: str
    gender: str
    description: str
    speed: float = 1.0
    temperature: float = 0.6
    is_custom: bool = False

class CustomModelManager:
    """Manages custom voice models."""
    
    def __init__(self):
        self.custom_models_dir = CUSTOM_MODELS_DIR
        self.config_file = CUSTOM_MODELS_CONFIG_FILE
        self._ensure_directories()
        self._load_config()
    
    def _ensure_directories(self):
        """Ensure custom models directory exists."""
        self.custom_models_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self):
        """Load custom models configuration."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                logging.error(f"Failed to load custom models config: {e}")
                self.config = {"models": []}
        else:
            self.config = {"models": []}
    
    def _save_config(self):
        """Save custom models configuration."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save custom models config: {e}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available TTS models."""
        if not TTS_AVAILABLE:
            return []
        
        try:
            tts = TTS()
            models = tts.list_models()
            return [str(m) for m in models]
        except Exception as e:
            logging.error(f"Failed to get available models: {e}")
            return []
    
    def get_custom_models(self) -> List[VoiceModel]:
        """Get list of custom voice models."""
        models = []
        for model_data in self.config.get("models", []):
            model = VoiceModel(
                name=model_data["name"],
                path=model_data["path"],
                gender=model_data["gender"],
                description=model_data["description"],
                speed=model_data.get("speed", 1.0),
                temperature=model_data.get("temperature", 0.6),
                is_custom=True
            )
            models.append(model)
        return models
    
    def add_custom_model(self, model_path: str, name: str, gender: str, 
                        description: str, speed: float = 1.0, 
                        temperature: float = 0.6) -> bool:
        """Add a custom voice model."""
        try:
            # Verify the model path exists
            if not os.path.exists(model_path):
                logging.error(f"Model path does not exist: {model_path}")
                return False
            
            # Add to configuration
            model_data = {
                "name": name,
                "path": model_path,
                "gender": gender,
                "description": description,
                "speed": speed,
                "temperature": temperature
            }
            
            self.config["models"].append(model_data)
            self._save_config()
            
            logging.info(f"Added custom model: {name} at {model_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to add custom model: {e}")
            return False
    
    def remove_custom_model(self, name: str) -> bool:
        """Remove a custom voice model."""
        try:
            # Find and remove the model
            models = self.config.get("models", [])
            for i, model in enumerate(models):
                if model["name"] == name:
                    del models[i]
                    self._save_config()
                    logging.info(f"Removed custom model: {name}")
                    return True
            
            logging.warning(f"Custom model not found: {name}")
            return False
            
        except Exception as e:
            logging.error(f"Failed to remove custom model: {e}")
            return False
    
    def open_model_download_page(self):
        """Open the Coqui TTS model download page."""
        url = "https://huggingface.co/models?search=tts&sort=downloads"
        try:
            webbrowser.open(url)
            return True
        except Exception as e:
            logging.error(f"Failed to open download page: {e}")
            return False
    
    def get_model_installation_instructions(self) -> str:
        """Get instructions for installing custom models."""
        return """
Custom Voice Model Installation Instructions:

1. Find TTS Models:
   - Visit: https://huggingface.co/models?search=tts&sort=downloads
   - Or search for "TTS" models on Hugging Face
   - Look for models with clear gender indicators (male/female)

2. Download Models:
   - Click on a model repository
   - Look for "Files and versions" section
   - Download model files (.pth, config.json, etc.)
   - Some models may be available via: git clone https://huggingface.co/[model-name]

3. Alternative Sources:
   - Coqui TTS GitHub: https://github.com/coqui-ai/TTS
   - Pre-trained models: https://github.com/coqui-ai/TTS/tree/dev#pre-trained-models
   - Community models: Check TTS community discussions

4. Install the Model:
   - Extract the downloaded files
   - Copy the model folder to: tts_models/custom/
   - The folder structure should be: tts_models/custom/your_model_name/

5. Add the Model to Ratatoskr:
   - Open Settings → Voice Settings → Custom Models
   - Click "Manage Custom Models"
   - Click "Add Model" and select your model directory
   - Enter the model details and test it

6. Test the Model:
   - Use the "Test Model" button to verify it works
   - Adjust speed and temperature as needed

Recommended Models:
- For Male Voice: Search for "male voice TTS" or "male speaker"
- For Female Voice: Search for "female voice TTS" or "female speaker"
- For Different Accents: Search for "british TTS", "australian TTS", etc.

Note: Some models may require specific TTS versions or additional dependencies.
        """
    
    def validate_model_path(self, model_path: str) -> bool:
        """Validate that a model path contains valid TTS model files."""
        if not os.path.exists(model_path):
            return False
        
        # Check for common TTS model files
        required_files = ["config.json", "model.pth"]
        optional_files = ["vocoder.pth", "speaker_encoders.pth"]
        
        # Check for at least one required file
        has_required = any(os.path.exists(os.path.join(model_path, f)) for f in required_files)
        
        # Check for model files
        has_model_files = any(f.endswith('.pth') for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f)))
        
        return has_required or has_model_files
    
    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """Get information about a model from its path."""
        info = {
            "name": os.path.basename(model_path),
            "path": model_path,
            "exists": os.path.exists(model_path),
            "files": []
        }
        
        if os.path.exists(model_path):
            try:
                files = os.listdir(model_path)
                info["files"] = [f for f in files if os.path.isfile(os.path.join(model_path, f))]
                
                # Try to read config.json if it exists
                config_file = os.path.join(model_path, "config.json")
                if os.path.exists(config_file):
                    try:
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                            info["config"] = config
                    except:
                        pass
                        
            except Exception as e:
                logging.error(f"Error reading model info: {e}")
        
        return info

# Global instance
custom_model_manager = CustomModelManager() 