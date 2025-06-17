# Ratatoskr AI Assistant

## Overview
Ratatoskr is a modular, cross-platform AI assistant for Linux and Windows. It features:
- Persistent memory and conversation history
- Fast, GPU-accelerated voice (TTS/STT) with gender selection
- Web search with browser fallback
- Gmail/calendar integration
- System automation: file management, LibreOffice, browser, screen, dictation, and more
- OS-aware: adapts all commands and UI to your system (KDE/GNOME/Windows)
- Modular, maintainable, and reproducible project structure

## Project Structure
```
ratatoskr/
├── src/ratatoskr/         # All main code modules (agent, tools, voice, etc.)
├── data/
│   ├── raw/               # Raw, unprocessed data
│   ├── processed/         # Cleaned/processed data
│   ├── external/          # External datasets, TTS models, etc.
│   └── interim/           # Intermediate data
├── notebooks/             # Jupyter notebooks for EDA, training, etc.
├── reports/
│   ├── figures/           # Plots, graphs, visualizations
│   └── results/           # Experiment results
├── logs/                  # All logs (training, inference, debug)
├── tests/                 # Unit and integration tests
├── scripts/               # Run scripts (training, inference, deployment)
├── config/                # YAML config files, credentials (not tracked)
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## OS Support & Adaptation
- **Linux:**
  - KDE/GNOME auto-detection
  - Uses correct terminal, file manager, text editor, calculator, settings, etc.
  - Scans `.desktop` files for all menu apps (launch any app by name)
- **Windows:**
  - Uses correct system apps (cmd, explorer, notepad, calc, etc.)
  - Can prompt for permission to scan for `.exe` or let user set custom app paths
- All system features (file management, LibreOffice, browser, screen, dictation, etc.) are fully OS-adaptive.

## Getting Started
1. **Clone the repo:**
   ```sh
   git clone <your-fork-or-main-url>
   cd ratatoskr
   ```
2. **Install dependencies:**
   ```sh
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
3. **Download required models/data:**
   - **Voice models:**
     - Download from [Hugging Face TTS models](https://huggingface.co/collections/tts-models)
     - Place in `data/external/tts_models/`
   - **Chroma DB:**
     - Place in `data/processed/chroma_db/` if needed
   - **Other large files:**
     - Not tracked by git. Download as needed per instructions in the docs.
4. **Configure:**
   - Edit `config/params.yaml` for all system, model, and integration settings
   - Place any credentials (Gmail, API keys) in `config/credentials.yaml` (not tracked by git)
5. **Run the app:**
   ```sh
   python src/ratatoskr/main.py
   ```

## Usage Examples
- "Open terminal" (launches correct terminal for your OS)
- "Open file manager" (dolphin, nautilus, or explorer)
- "Open calculator", "Open text editor", "Open settings"
- "Open [any app name]" (fuzzy-matches and launches any app from your menu)
- All features (voice, memory, web search, Gmail, LibreOffice, etc.) are available and OS-adaptive

## .gitignore Policy
- **Tracked:** Only code, scripts, configs (no secrets), and small sample data
- **Ignored:**
  - All large files (voice models, datasets, Chroma DB, etc.)
  - All logs, outputs, and user data
  - All credentials and secrets
- **How to get required files:**
  - Download voice models and place in `data/external/tts_models/`
  - Download any other required data as per the docs
  - Place credentials in `config/credentials.yaml` (never commit this file)

## Contributing & Development
- All code is modular and follows best practices for maintainability and reproducibility
- Add new features in `src/ratatoskr/` and update tests in `tests/`
- Use YAML for all configuration
- See `DEVELOPMENT.md` for more details

## License
See `LICENSE` for details.

## 🌟 Features

### 🤖 AI Capabilities
- **Local AI Processing**: Powered by Ollama with support for various models (Llama, Mistral, etc.)
- **Multi-Modal Interaction**: Text, voice, and hybrid interaction modes
- **Memory System**: Persistent conversation history and long-term memory storage
- **Web Integration**: Real-time web search and browsing capabilities
- **Tool Integration**: Access to various tools for enhanced functionality

### 🎤 Voice Features
- **Text-to-Speech**: High-quality speech synthesis with multiple voice options
  - **Male Voice**: Fast Pitch model for clear, responsive male voice
  - **Female Voice**: Tacotron2 model for natural female voice
  - **Multi-Speaker**: VITS model with multiple speaker options
- **Speech-to-Text**: Real-time voice recognition using Whisper
- **Voice Configuration**: Adjustable speed, temperature, and gender selection
- **Streaming Playback**: Chunked audio for responsive voice interaction

### 📧 Gmail Integration
- **Email Management**: Check and read recent emails
- **Calendar Integration**: Monitor calendar events and appointments
- **Smart Alerts**: Configurable notifications (15, 10, 5 minutes before events)
- **Daily Summaries**: Automated daily activity summaries
- **Background Monitoring**: Continuous calendar monitoring with voice alerts

### ⚙️ Configuration System
- **Voice Settings**: Choose male/female voice, adjust speed and quality
- **Ollama Configuration**: Customize server URL, model selection, and timeouts
- **Gmail Settings**: Configure email, authentication, and alert preferences
- **Persistent Settings**: All configurations saved and restored between sessions

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Ollama installed and running
- Microphone and speakers
- (Optional) Gmail account for calendar integration

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ratatoskr
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama** (if not already installed):
   ```bash
   # Follow instructions at https://ollama.ai
   ollama pull llama3.1:8b
   ```

5. **Run the application**:
   ```bash
   python main.py
   ```

## 🎛️ Configuration

### Voice Settings
Access voice configuration through **Settings** (Ctrl+,):

- **Voice Gender**: Choose between Male and Female voices
- **TTS Model**: Select from Fast Pitch (fast), Tacotron2 (quality), or VITS (multi-speaker)
- **Speech Speed**: Adjust playback speed (50% - 200%)
- **Temperature**: Control voice variation (10% - 100%)

### Ollama Configuration
Configure your AI model settings:

- **Model Name**: Select your preferred Ollama model (e.g., `llama3.1:8b`)
- **Server URL**: Set Ollama server address (default: `http://127.0.0.1:11434`)
- **Timeout**: Configure request timeout (10-120 seconds)

### Gmail Integration
Set up Gmail and Calendar integration:

1. **Enable Gmail Integration**: Check the box to enable
2. **Email Address**: Enter your Gmail address
3. **App Password**: Use Gmail App Password (not regular password)
4. **Calendar Alerts**: Enable/disable event notifications
5. **Alert Times**: Choose when to receive alerts (15, 10, 5 minutes before)
6. **Daily Summary**: Enable automated daily summaries
7. **Summary Time**: Set when to receive daily summaries

#### Gmail Setup Instructions
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Gmail API and Google Calendar API
4. Create OAuth 2.0 credentials
5. Download `credentials.json` and place in project directory
6. Configure settings in the application

## 🎯 Usage

### Basic Interaction
1. **Start the application**: `python main.py`
2. **Choose interaction mode**:
   - **Hybrid**: Use both text and voice
   - **Voice Only**: Voice-only interaction
   - **Text Only**: Text-only interaction
3. **Start conversing**: Type or speak your questions

### Voice Commands
- Click **Listen** button or use voice mode for hands-free interaction
- Speak naturally - the system will transcribe and respond
- Voice responses will play automatically in voice/hybrid modes

### Gmail Features
- **Calendar Alerts**: Receive voice notifications before events
- **Daily Summary**: Get morning overview of your day
- **Email Checking**: Ask about recent emails
- **Event Management**: Query calendar events and schedules

### Memory and Context
- **Conversation History**: All conversations are automatically saved
- **Long-term Memory**: Important information is stored for future reference
- **Context Awareness**: The AI remembers previous conversations and can reference them

## 🔧 Advanced Configuration

### GPU Acceleration
For faster TTS processing, enable GPU acceleration:

```bash
# For NVIDIA GPUs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For AMD GPUs (ROCm)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```

### Custom Ollama Models
Add custom models to Ollama:

```bash
# Pull additional models
ollama pull mistral:7b
ollama pull codellama:7b

# Use in settings
# Model Name: mistral:7b
```

### Voice Model Customization
Advanced voice settings in `config.json`:

```json
{
  "voice": {
    "gender": "male",
    "model": "tts_models/en/ljspeech/fast_pitch",
    "speed": 1.2,
    "temperature": 0.5
  }
}
```

## 🧪 Testing

Run the test suite to verify all features:

```bash
# Test new features
python test_new_features.py

# Run comprehensive tests
python -m pytest tests/
```

## 📁 Project Structure

```
ratatoskr/
├── main.py                 # Main application entry point
├── config.py              # Configuration management
├── settings_dialog.py     # Settings GUI
├── gmail_integration.py   # Gmail and Calendar integration
├── voice/
│   ├── text_to_speech.py  # TTS functionality
│   └── speech_to_text.py  # STT functionality
├── memory/
│   └── conversation_manager.py  # Memory management
├── tools/
│   ├── web_search.py      # Web search tools
│   └── browser_tool.py    # Web browsing tools
├── agent/
│   └── execute.py         # AI agent execution
├── tests/                 # Test suite
└── requirements.txt       # Dependencies
```

## 🐛 Troubleshooting

### Common Issues

**Voice not working**:
- Check microphone permissions
- Verify PyAudio installation
- Test with `python test_voice.py`

**Ollama connection failed**:
- Ensure Ollama is running: `ollama serve`
- Check server URL in settings
- Verify model is installed: `ollama list`

**Gmail integration issues**:
- Verify `credentials.json` is in project directory
- Check Gmail API is enabled in Google Cloud Console
- Ensure App Password is used (not regular password)

**Performance issues**:
- Enable GPU acceleration for TTS
- Use faster models (Fast Pitch for voice, smaller LLM models)
- Adjust chunk sizes in settings

### Logs and Debugging
- Check application logs in console output
- Enable debug logging in `logging_config.py`
- Use test scripts to isolate issues

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ollama** for local AI model serving
- **Coqui TTS** for high-quality text-to-speech
- **LangChain** for AI agent framework
- **PyQt6** for the user interface
- **Google APIs** for Gmail integration

---

**Ratatoskr** - Your intelligent, voice-enabled AI companion! 🐿️✨

## 🖥️ OS-Aware System Initialization and Application Launching

Ratatoskr now features a comprehensive system controller that:

- **Detects your OS and desktop environment** (Linux or Windows, KDE/GNOME, etc.)
- **Creates all required directories** (`~/ratatoskr/data`, `~/ratatoskr/logs`, etc.) on startup if missing
- **Adapts all system commands** (terminal, file manager, text editor, calculator, settings, app launching) to your OS and environment
- **On Linux:**
  - Scans `.desktop` files in `/usr/share/applications`, `/usr/local/share/applications`, and `~/.local/share/applications` to build a menu of launchable apps
  - Uses the correct app for your desktop (e.g., `konsole`/`dolphin` on KDE, `gnome-terminal`/`nautilus` on GNOME)
- **On Windows:**
  - Uses common app locations and can prompt the user for permission to scan or to manually set `.exe` paths for custom apps
  - Adapts all system commands to Windows equivalents (e.g., `cmd`, `explorer`, `notepad`, etc.)

### Usage Examples

- "Open terminal" → launches the correct terminal for your OS/desktop
- "Open file manager" → launches `dolphin`, `nautilus`, or `explorer` as appropriate
- "Open calculator", "Open text editor", "Open settings" → always uses the right app
- "Open [any app name]" → fuzzy-matches and launches any app from your system menu

All system features (file management, LibreOffice, browser, screen, dictation, etc.) are now fully OS-adaptive and robust.

---
