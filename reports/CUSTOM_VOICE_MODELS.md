# Custom Voice Models Guide

Ratatoskr AI Assistant now supports custom voice models, allowing you to use different voices, accents, and languages for a more personalized experience.

## Overview

The custom voice model system allows you to:
- Download and install custom TTS models
- Manage multiple voice models
- Test and configure voice settings
- Use different voices for different purposes

## Getting Started

### 1. Access Custom Models

1. Open Ratatoskr AI Assistant
2. Go to **Settings** → **Voice Settings**
3. Click on the **Custom Models** tab
4. Click **Manage Custom Models**

### 2. Find Models

The system provides several ways to find models:

#### Option A: Model Recommendations
1. Click **Show Model Recommendations** in the Custom Models dialog
2. Browse curated models organized by gender and quality
3. Click **Download** next to any model to open its page

#### Option B: Search Hugging Face
1. Click **Open Model Download Page** to search Hugging Face
2. Use search terms like:
   - "male voice TTS"
   - "female voice TTS" 
   - "british TTS"
   - "australian TTS"
   - "fastspeech2"
   - "your-tts"

#### Option C: Direct Links
Visit these specific model repositories:
- **YourTTS (Multi-language)**: https://huggingface.co/coqui/your-tts
- **Coqui TTS v2**: https://huggingface.co/coqui/tts-v2
- **FastSpeech2 Models**: https://huggingface.co/models?search=fastspeech2

### 3. Download Models

#### For Git Clone Models:
```bash
git clone https://huggingface.co/coqui/your-tts
git clone https://huggingface.co/coqui/tts-v2
```

#### For Direct Download Models:
1. Visit the model page on Hugging Face
2. Look for "Files and versions" section
3. Download model files (.pth, config.json, etc.)

### 4. Install Models

1. Extract the downloaded files
2. Copy the model folder to: `tts_models/custom/`
3. The folder structure should be: `tts_models/custom/your_model_name/`

### 5. Add to Ratatoskr

1. In the Custom Models dialog, click **Add Model**
2. Select the model directory you copied
3. Fill in the model details:
   - **Name**: A descriptive name for the model
   - **Gender**: male, female, or neutral
   - **Description**: Brief description of the voice
   - **Speed**: Speech speed (50-200%)
   - **Temperature**: Voice variation (10-100%)
4. Click **Test Model** to verify it works
5. Close the dialog to save

## Recommended Models

### Quick Start Models (Beginner Friendly)
- **YourTTS**: Multi-language, high quality, both genders
- **Coqui TTS v2**: English, high quality, both genders

### For Male Voices
- **YourTTS - Male Voice**: Multi-language, very high quality
- **Coqui TTS - VCTK Male**: English, high quality
- **FastSpeech2 - Male**: English, good quality, smaller size

### For Female Voices
- **YourTTS - Female Voice**: Multi-language, very high quality
- **Coqui TTS - LJSpeech**: English, high quality
- **FastSpeech2 - Female**: English, good quality, smaller size

### For Different Accents
- Search for "british TTS", "australian TTS", "indian TTS"
- Look for models with country/region indicators

## Model Management

### Viewing Models
- All custom models are listed in the Custom Models dialog
- Each model shows its name, gender, and description
- Click on a model to view its details

### Testing Models
- Select a model from the list
- Click **Test Model** to hear a sample
- The test will speak: "Hello, this is a test of the custom model [name]. It should sound natural and clear."

### Removing Models
- Select a model from the list
- Click **Remove Model**
- This only removes it from Ratatoskr's configuration
- The model files remain on disk

## Technical Details

### Model Requirements
Custom models must contain:
- Model files (`.pth` files)
- Configuration files (`config.json`)
- Optional: Vocoder files (`vocoder.pth`)

### File Structure
```
tts_models/custom/
├── your_model_name/
│   ├── config.json
│   ├── model.pth
│   ├── vocoder.pth (optional)
│   └── other files...
└── models.json (configuration)
```

### Configuration File
The `models.json` file stores custom model configurations:
```json
{
  "models": [
    {
      "name": "My Custom Model",
      "path": "tts_models/custom/my_model",
      "gender": "male",
      "description": "A custom male voice",
      "speed": 1.0,
      "temperature": 0.6
    }
  ]
}
```

## Troubleshooting

### Model Not Working
1. Check that the model directory contains valid TTS files
2. Ensure the model is compatible with your TTS version
3. Try testing the model directly with TTS library
4. Check the logs for error messages

### Model Sounds Wrong
1. Adjust the **Speed** setting (slower = clearer)
2. Adjust the **Temperature** setting (lower = more consistent)
3. Try different models for better quality

### Model Not Loading
1. Verify the model path is correct
2. Check file permissions
3. Ensure all required files are present
4. Restart Ratatoskr after adding models

### No Models Found
1. Use the **Show Model Recommendations** feature
2. Search Hugging Face with specific terms
3. Check the model discovery system for curated lists
4. Visit the Coqui TTS GitHub repository

## Advanced Usage

### Using Custom Models in Code
```python
from voice.text_to_speech import speak_sync

# Use a custom model
speak_sync(
    "Hello, this is a test",
    model_path="tts_models/custom/my_model",
    speed=1.0,
    temperature=0.6
)
```

### Batch Model Testing
```python
from voice.custom_models import custom_model_manager

# Test all custom models
for model in custom_model_manager.get_custom_models():
    print(f"Testing {model.name}...")
    speak_sync("Test", model_path=model.path)
```

## Community Models

### Where to Find Models
- **Hugging Face**: https://huggingface.co/models?search=tts&sort=downloads
- **Coqui TTS GitHub**: https://github.com/coqui-ai/TTS
- **Pre-trained Models**: https://github.com/coqui-ai/TTS/tree/dev#pre-trained-models
- **Community Forums**: Check TTS community discussions

### Model Quality Tips
- Look for models with high ratings and reviews
- Test models before committing to them
- Consider model size vs. quality trade-offs
- Some models may require specific TTS versions
- Start with smaller models for testing

## Support

If you encounter issues with custom models:
1. Check the application logs for error messages
2. Verify model compatibility with your TTS version
3. Test models with the TTS library directly
4. Use the model recommendations feature
5. Consult the Coqui TTS documentation

## Future Enhancements

Planned features for custom voice models:
- Automatic model downloading
- Voice cloning capabilities
- Multi-language support
- Real-time voice switching
- Voice quality optimization
- Model marketplace integration

---

For more information, visit the [Coqui TTS documentation](https://docs.coqui.ai/) or check the [Ratatoskr project repository](https://github.com/your-repo/ratatoskr). 