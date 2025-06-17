# Ratatoskr AI Assistant - Development Guide

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Setup and Installation](#setup-and-installation)
4. [Development Environment](#development-environment)
5. [Testing](#testing)
6. [Code Quality](#code-quality)
7. [Contributing](#contributing)
8. [Troubleshooting](#troubleshooting)

## Project Overview

Ratatoskr is an intelligent AI assistant that combines natural language processing, speech recognition, computer vision, and external tool integration to provide a comprehensive AI experience.

### Key Features

- **Natural Language Processing**: Powered by Ollama with local model support
- **Speech Processing**: Text-to-speech and speech-to-text capabilities
- **Memory Management**: Short-term and long-term memory with vector storage
- **Tool Integration**: Web search, browser automation, and external APIs
- **Computer Vision**: UI element localization and screen interaction
- **Modular Architecture**: Extensible component-based design

## Architecture

### Core Components

```
ratatoskr/
├── agent/           # Decision making and execution
├── memory/          # Memory management systems
├── voice/           # Speech processing
├── tools/           # External tool integrations
├── vision/          # Computer vision components
├── llm/             # Language model integration
├── validator/       # Content validation
└── config.py        # Configuration management
```

### Data Flow

1. **Input Processing**: User input (text/voice) → Speech-to-text → Input analysis
2. **Decision Making**: Policy engine analyzes input and selects appropriate actions
3. **Tool Execution**: Selected tools perform requested operations
4. **Memory Integration**: Results stored in short-term and long-term memory
5. **Response Generation**: LLM generates contextual responses
6. **Output Delivery**: Text-to-speech or text output to user

## Setup and Installation

### Prerequisites

- Python 3.8+
- Ollama (for local LLM support)
- Audio hardware (for voice features)
- GPU (optional, for enhanced performance)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/ratatoskr.git
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

4. **Install Ollama**:
   ```bash
   # Follow instructions at https://ollama.ai
   ollama pull llama3.1:8b
   ```

5. **Configure environment**:
   ```bash
   # Copy and edit configuration
   cp .env.example .env
   # Edit .env with your settings
   ```

### Configuration

The application uses a comprehensive configuration system with environment variable support:

```python
# Example configuration
RATATOSKR_MODEL_NAME=llama3.1:8b
RATATOSKR_TEMPERATURE=0.7
RATATOSKR_TTS_MODEL=tts_models/en/ljspeech/tacotron2-DDC
RATATOSKR_STT_MODEL=small.en
RATATOSKR_LOG_LEVEL=INFO
```

## Development Environment

### IDE Setup

Recommended IDE: **VS Code** with Python extensions

1. **Install VS Code extensions**:
   - Python
   - Pylance
   - Python Test Explorer
   - GitLens

2. **Configure Python interpreter**:
   - Select the virtual environment Python interpreter
   - Enable type checking with Pylance

3. **Code formatting**:
   ```bash
   pip install black isort
   ```

### Development Workflow

1. **Create feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes**:
   - Follow coding standards
   - Add tests for new functionality
   - Update documentation

3. **Run tests**:
   ```bash
   python tests/run_tests.py --suite all
   ```

4. **Code quality checks**:
   ```bash
   black .
   isort .
   flake8 .
   ```

5. **Commit changes**:
   ```bash
   git add .
   git commit -m "feat: add new feature"
   git push origin feature/your-feature-name
   ```

## Testing

### Test Structure

```
tests/
├── test_memory_long_term.py    # Memory system tests
├── test_agent.py               # Agent component tests
├── test_tools.py               # Tool integration tests
├── test_voice.py               # Voice processing tests
└── run_tests.py                # Test runner
```

### Running Tests

1. **Run all tests**:
   ```bash
   python tests/run_tests.py --suite all
   ```

2. **Run specific test suite**:
   ```bash
   python tests/run_tests.py --suite memory
   python tests/run_tests.py --suite agent
   python tests/run_tests.py --suite tools
   python tests/run_tests.py --suite voice
   ```

3. **Run performance tests**:
   ```bash
   python tests/run_tests.py --performance
   ```

4. **Run integration tests**:
   ```bash
   python tests/run_tests.py --integration
   ```

5. **Generate test report**:
   ```bash
   python tests/run_tests.py --suite all --report --output test_report.txt
   ```

### Test Categories

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **Performance Tests**: Test system performance and scalability
- **Error Handling Tests**: Test error conditions and recovery

### Writing Tests

Follow these guidelines when writing tests:

1. **Test naming**: Use descriptive test method names
2. **Test isolation**: Each test should be independent
3. **Mocking**: Use mocks for external dependencies
4. **Coverage**: Aim for high test coverage
5. **Documentation**: Document complex test scenarios

Example test structure:

```python
class TestMyComponent(unittest.TestCase):
    """Test suite for MyComponent."""
    
    def setUp(self):
        """Set up test environment."""
        self.component = MyComponent()
    
    def test_successful_operation(self):
        """Test successful operation."""
        result = self.component.operation("test_input")
        self.assertEqual(result, "expected_output")
    
    def test_error_handling(self):
        """Test error handling."""
        with self.assertRaises(ValueError):
            self.component.operation("")
```

## Code Quality

### Coding Standards

1. **PEP 8 Compliance**: Follow Python style guidelines
2. **Type Hints**: Use type annotations for all functions
3. **Docstrings**: Document all public functions and classes
4. **Error Handling**: Implement proper error handling
5. **Logging**: Use structured logging throughout

### Code Formatting

```bash
# Format code with Black
black .

# Sort imports with isort
isort .

# Check code style with flake8
flake8 .
```

### Type Checking

```bash
# Run type checking with mypy
mypy .
```

### Linting

```bash
# Run comprehensive linting
flake8 . --max-line-length=88 --extend-ignore=E203,W503
```

## Contributing

### Contribution Guidelines

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Add tests**
5. **Update documentation**
6. **Submit a pull request**

### Pull Request Process

1. **Description**: Provide clear description of changes
2. **Tests**: Ensure all tests pass
3. **Documentation**: Update relevant documentation
4. **Code Review**: Address review comments
5. **Merge**: Maintainer merges after approval

### Commit Message Format

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Maintenance tasks

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] Error handling is implemented
- [ ] Logging is appropriate
- [ ] Performance considerations addressed

## Troubleshooting

### Common Issues

#### Ollama Connection Issues

```bash
# Check Ollama status
ollama list

# Restart Ollama service
sudo systemctl restart ollama

# Test connection
curl http://localhost:11434/api/tags
```

#### Audio Issues

```bash
# Check audio devices
python -c "import pyaudio; p = pyaudio.PyAudio(); print(p.get_device_count())"

# Test microphone
python -c "from voice.speech_to_text import test_microphone; test_microphone()"
```

#### Memory Issues

```bash
# Check ChromaDB status
python -c "import chromadb; client = chromadb.PersistentClient()"

# Clear memory if needed
python -c "from memory.long_term import clear_memory; clear_memory()"
```

#### Performance Issues

1. **Check system resources**:
   ```bash
   htop
   nvidia-smi  # If using GPU
   ```

2. **Monitor logs**:
   ```bash
   tail -f application.log
   ```

3. **Profile code**:
   ```bash
   python -m cProfile -o profile.stats main.py
   ```

### Debug Mode

Enable debug logging:

```bash
export RATATOSKR_LOG_LEVEL=DEBUG
python main.py
```

### Environment Variables

Key environment variables for debugging:

```bash
RATATOSKR_LOG_LEVEL=DEBUG
RATATOSKR_VERBOSE=true
RATATOSKR_DEBUG=true
```

## Performance Optimization

### Memory Management

1. **Vector Database**: Use ChromaDB for efficient memory storage
2. **Caching**: Implement caching for frequently accessed data
3. **Cleanup**: Regular cleanup of temporary files and logs

### Processing Optimization

1. **Async Operations**: Use async/await for I/O operations
2. **Chunking**: Process large texts in chunks
3. **Parallel Processing**: Use threading for independent operations

### Resource Management

1. **GPU Usage**: Optimize GPU memory usage for ML models
2. **CPU Usage**: Use multiprocessing for CPU-intensive tasks
3. **Network**: Implement connection pooling for external APIs

## Security Considerations

### Data Privacy

1. **Local Processing**: Keep sensitive data local when possible
2. **Encryption**: Encrypt stored data
3. **Access Control**: Implement proper access controls

### API Security

1. **Rate Limiting**: Implement rate limiting for external APIs
2. **Authentication**: Use secure authentication methods
3. **Input Validation**: Validate all user inputs

### Model Security

1. **Model Validation**: Validate model outputs
2. **Content Filtering**: Implement content filtering
3. **Bias Detection**: Monitor for model bias

## Deployment

### Production Setup

1. **Environment**: Use production-grade environment
2. **Monitoring**: Implement comprehensive monitoring
3. **Backup**: Regular backup of data and configuration
4. **Security**: Implement security best practices

### Containerization

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

### CI/CD Pipeline

Example GitHub Actions workflow:

```yaml
name: CI/CD Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          python tests/run_tests.py --suite all
```

## Support and Community

### Getting Help

1. **Documentation**: Check this guide and README.md
2. **Issues**: Create GitHub issues for bugs
3. **Discussions**: Use GitHub Discussions for questions
4. **Community**: Join our community channels

### Reporting Bugs

When reporting bugs, include:

1. **Environment**: OS, Python version, dependencies
2. **Steps**: Clear steps to reproduce
3. **Expected vs Actual**: Expected vs actual behavior
4. **Logs**: Relevant log files
5. **Screenshots**: If applicable

### Feature Requests

When requesting features:

1. **Description**: Clear description of the feature
2. **Use Case**: Explain the use case
3. **Implementation**: Suggest implementation approach
4. **Priority**: Indicate priority level

## OS-Aware System Controller Architecture

### Overview
The system controller is responsible for:
- Detecting the OS and desktop environment (Linux/Windows, KDE/GNOME, etc.)
- Creating all required directories (`~/ratatoskr/data`, `~/ratatoskr/logs`, etc.) on startup
- Adapting all system commands (terminal, file manager, text editor, calculator, settings, app launching) to the OS and environment
- Scanning for available applications and providing robust, fuzzy-matched launching

### Linux
- Scans `.desktop` files in `/usr/share/applications`, `/usr/local/share/applications`, and `~/.local/share/applications` to build a menu of launchable apps
- Uses the correct app for the desktop (e.g., `konsole`/`dolphin` for KDE, `gnome-terminal`/`nautilus` for GNOME)
- All system commands are mapped to the appropriate Linux tools

### Windows
- Uses common app locations and can prompt the user for permission to scan or to manually set `.exe` paths for custom apps
- All system commands are mapped to Windows equivalents (e.g., `cmd`, `explorer`, `notepad`, etc.)

### Directory Creation
- On startup, the following directories are created if missing:
  - `~/ratatoskr`
  - `~/ratatoskr/data`
  - `~/ratatoskr/logs`
  - `~/ratatoskr/screenshots`
  - `~/ratatoskr/documents`
  - `~/ratatoskr/temp`

### Application Launching
- The `launch_application` function supports launching by generic name (`terminal`, `file_manager`, etc.) and will use the correct app for the OS and desktop environment
- Fuzzy-matching is used to launch any app from the system menu

### Developer Notes
- To extend or customize app mappings, update the mappings in the system controller or add new logic for additional OSes/desktops
- For Windows, you can add a GUI dialog to request permission to scan for `.exe` files or let users manually specify paths
- For Linux, you can add more directories to scan for `.desktop` files if needed

---

For more information, see the main [README.md](README.md) file. 