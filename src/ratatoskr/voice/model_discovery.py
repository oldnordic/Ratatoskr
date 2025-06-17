"""
Model Discovery for Ratatoskr AI Assistant.

This module provides curated lists of downloadable TTS models and
helps users find suitable models for their needs.
"""

import webbrowser
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ModelInfo:
    """Information about a downloadable TTS model."""
    name: str
    url: str
    description: str
    gender: str
    language: str
    size: str
    quality: str
    download_method: str

class ModelDiscovery:
    """Helps users discover and download TTS models."""
    
    def __init__(self):
        self.curated_models = self._get_curated_models()
    
    def _get_curated_models(self) -> List[ModelInfo]:
        """Get a curated list of downloadable TTS models."""
        return [
            # Male Voice Models
            ModelInfo(
                name="Coqui TTS - Male Speaker (VCTK)",
                url="https://huggingface.co/coqui/tts-v2",
                description="High-quality male voice from VCTK dataset",
                gender="male",
                language="English",
                size="~500MB",
                quality="High",
                download_method="git clone"
            ),
            ModelInfo(
                name="YourTTS - Male Voice",
                url="https://huggingface.co/coqui/your-tts",
                description="YourTTS model with male voice capabilities",
                gender="male",
                language="Multi-language",
                size="~1GB",
                quality="Very High",
                download_method="git clone"
            ),
            ModelInfo(
                name="FastSpeech2 - Male",
                url="https://huggingface.co/models?search=fastspeech2+male",
                description="FastSpeech2 models with male voices",
                gender="male",
                language="English",
                size="~200MB",
                quality="Good",
                download_method="Direct download"
            ),
            
            # Female Voice Models
            ModelInfo(
                name="Coqui TTS - LJSpeech (Female)",
                url="https://huggingface.co/coqui/tts-v2",
                description="High-quality female voice from LJSpeech dataset",
                gender="female",
                language="English",
                size="~500MB",
                quality="High",
                download_method="git clone"
            ),
            ModelInfo(
                name="YourTTS - Female Voice",
                url="https://huggingface.co/coqui/your-tts",
                description="YourTTS model with female voice capabilities",
                gender="female",
                language="Multi-language",
                size="~1GB",
                quality="Very High",
                download_method="git clone"
            ),
            ModelInfo(
                name="FastSpeech2 - Female",
                url="https://huggingface.co/models?search=fastspeech2+female",
                description="FastSpeech2 models with female voices",
                gender="female",
                language="English",
                size="~200MB",
                quality="Good",
                download_method="Direct download"
            ),
            
            # Accent Models
            ModelInfo(
                name="British Accent TTS",
                url="https://huggingface.co/models?search=british+tts",
                description="British accent voice models",
                gender="both",
                language="English (British)",
                size="~300MB",
                quality="Good",
                download_method="Direct download"
            ),
            ModelInfo(
                name="Australian Accent TTS",
                url="https://huggingface.co/models?search=australian+tts",
                description="Australian accent voice models",
                gender="both",
                language="English (Australian)",
                size="~300MB",
                quality="Good",
                download_method="Direct download"
            ),
        ]
    
    def get_models_by_gender(self, gender: str) -> List[ModelInfo]:
        """Get models filtered by gender."""
        if gender == "both":
            return self.curated_models
        return [model for model in self.curated_models if model.gender == gender]
    
    def get_models_by_language(self, language: str) -> List[ModelInfo]:
        """Get models filtered by language."""
        return [model for model in self.curated_models if language.lower() in model.language.lower()]
    
    def open_model_page(self, model: ModelInfo) -> bool:
        """Open the model's download page in browser."""
        try:
            webbrowser.open(model.url)
            return True
        except Exception as e:
            print(f"Failed to open model page: {e}")
            return False
    
    def get_download_instructions(self, model: ModelInfo) -> str:
        """Get specific download instructions for a model."""
        if "git clone" in model.download_method.lower():
            return f"""
Download Instructions for {model.name}:

1. Open terminal/command prompt
2. Navigate to your desired download directory
3. Run: git clone {model.url}
4. The model files will be downloaded to a new folder
5. Copy the model folder to: tts_models/custom/
6. Add the model in Ratatoskr's Custom Models dialog

Model Details:
- Size: {model.size}
- Quality: {model.quality}
- Language: {model.language}
- Gender: {model.gender}
            """
        else:
            return f"""
Download Instructions for {model.name}:

1. Visit: {model.url}
2. Look for "Files and versions" section
3. Download the model files (.pth, config.json, etc.)
4. Extract the downloaded files
5. Copy the model folder to: tts_models/custom/
6. Add the model in Ratatoskr's Custom Models dialog

Model Details:
- Size: {model.size}
- Quality: {model.quality}
- Language: {model.language}
- Gender: {model.gender}
            """
    
    def get_general_instructions(self) -> str:
        """Get general instructions for finding and downloading models."""
        return """
Finding and Downloading TTS Models:

1. Search on Hugging Face:
   - Visit: https://huggingface.co/models?search=tts&sort=downloads
   - Use search terms like: "male voice TTS", "female voice TTS", "british TTS"
   - Look for models with high download counts and good ratings

2. Alternative Sources:
   - Coqui TTS GitHub: https://github.com/coqui-ai/TTS
   - Pre-trained models: https://github.com/coqui-ai/TTS/tree/dev#pre-trained-models
   - Community forums and discussions

3. Download Methods:
   - Git Clone: Use "git clone [repository-url]" for full models
   - Direct Download: Download individual files from model pages
   - Model Hub: Use Hugging Face's model hub interface

4. Model Requirements:
   - Look for models with .pth files (model weights)
   - Check for config.json (model configuration)
   - Ensure compatibility with your TTS version

5. Installation:
   - Extract downloaded files
   - Copy to tts_models/custom/[model-name]/
   - Add in Ratatoskr's Custom Models dialog
   - Test before using

Tips:
- Start with smaller models for testing
- Check model documentation for requirements
- Test models before committing to them
- Some models may require specific TTS versions
        """
    
    def get_quick_start_models(self) -> List[ModelInfo]:
        """Get a list of models recommended for quick start."""
        return [
            model for model in self.curated_models 
            if "VCTK" in model.name or "YourTTS" in model.name
        ]

# Global instance
model_discovery = ModelDiscovery() 