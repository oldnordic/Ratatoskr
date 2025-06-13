# Project Ratatoskr: A Local AI Voice Assistant

A modular, local-first AI assistant built in Python. Ratatoskr leverages local large language models (via Ollama), speech-to-text (Whisper), and text-to-speech (Piper) to provide a private and extensible conversational AI experience.

## Core Features

-   **Modular Architecture:** Code is organized into distinct modules for the GUI, LLM client, voice I/O, and memory, making it easy to extend and maintain.
-   **Graphical User Interface:** A clean and responsive UI built with PyQt5, featuring a conversation view and interactive controls.
-   **Fully Local AI:** All core AI processing happens on your machine.
    -   **LLM Integration:** Connects to any model served by **Ollama**.
    -   **Speech-to-Text:** Uses OpenAI's **Whisper** for high-quality, offline transcription.
    -   **Text-to-Speech:** Uses **Piper TTS** for a natural, high-quality local voice.
-   **Long-Term Memory:** Implements a "learning" capability by using Sentence Transformers (PyTorch) to create embeddings for conversation snippets, storing and retrieving them from a **ChromaDB** vector database.
-   **Multiple Interaction Modes:** Easily switch between three modes via the UI:
    -   **Hybrid Mode:** Full text and voice input/output.
    -   **Voice-Only Mode:** For hands-free interaction.
    -   **Text-Only Mode:** For silent, classic chatbot interaction.

## Setup and Installation

### Prerequisites
-   Python 3.10+
-   `git`
-   **Ollama:** Must be installed and running on your local machine.
-   **System Dependencies:**
    -   `ffmpeg` (required by Whisper)
    -   `portaudio` (required by sounddevice)

    ```bash
    # On Debian/Ubuntu:
    sudo apt update && sudo apt install ffmpeg libportaudio2 libportaudiocpp0 portaudio19-dev

    # On Arch Linux / CachyOS:
    sudo pacman -S ffmpeg portaudio

    # On Windows (via Chocolatey - Recommended for ease of use):
    # Install Chocolatey from [https://chocolatey.org/install](https://chocolatey.org/install)
    choco install ffmpeg portaudio

    # On Windows (Manual Installation):
    # - Download ffmpeg from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) and add it to your system PATH.
    # - Download PortAudio from [http://www.portaudio.com/download.html](http://www.portaudio.com/download.html) and follow its build instructions.
    #   Ensure the compiled libraries (e.g., portaudio_x64.dll) are in your system PATH or Python's environment.
    ```

### Installation Steps

1.  **Clone the Repository:**
    ```bash
    # Replace with your actual repository URL
    git clone [https://github.com/YOUR_USERNAME/ratatoskr.git](https://github.com/YOUR_USERNAME/ratatoskr.git)
    cd ratatoskr
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    # On Linux/macOS:
    python3 -m venv venv
    source venv/bin/activate

    # On Windows (Command Prompt):
    python -m venv venv
    venv\Scripts\activate

    # On Windows (PowerShell):
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```

3.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download TTS Voice Model:**
    -   Go to the [Piper voice samples page](https://rhasspy.github.io/piper-samples/).
    -   Find a voice you like (e.g., `en_US-lessac-medium`).
    -   Create a folder named `tts_models` in the project root.
    -   Download both the `.onnx` and `.onnx.json` files for your chosen voice and place them in the `tts_models` folder.

## Running the Application

With your virtual environment active and Ollama running, start the assistant:

```bash
python main.py
