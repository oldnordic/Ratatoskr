"""
Configuration management for Ratatoskr AI Assistant.

This module handles all configuration settings including:
- AI model settings
- Voice preferences (male/female)
- Ollama server settings
- Gmail integration
- Calendar alert settings
"""

import json
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

# Default configuration
DEFAULT_CONFIG = {
    "ai_model": {
        "name": "llama3.1:8b",
        "ollama_server": "http://127.0.0.1:11434",
        "timeout": 30
    },
    "voice": {
        "gender": "male",  # "male" or "female"
        "model": "tts_models/en/ljspeech/fast_pitch",
        "speed": 1.2,
        "temperature": 0.5
    },
    "gmail": {
        "enabled": False,
        "email": "",
        "app_password": "",  # Gmail App Password
        "calendar_alerts": {
            "enabled": True,
            "alert_times": [15, 10, 5],  # minutes before event
            "daily_summary": True,
            "summary_time": "08:00"  # daily summary time
        }
    },
    "ui": {
        "theme": "dark",
        "window_size": [800, 600],
        "auto_start_listening": False
    }
}

@dataclass
class VoiceConfig:
    """Voice configuration settings."""
    gender: str = "male"
    model: str = "tts_models/en/ljspeech/fast_pitch"
    speed: float = 1.2
    temperature: float = 0.5

@dataclass
class OllamaConfig:
    """Ollama server configuration."""
    name: str = "llama3.1:8b"
    server_url: str = "http://127.0.0.1:11434"
    timeout: int = 30

@dataclass
class GmailConfig:
    """Gmail integration configuration."""
    enabled: bool = False
    email: str = ""
    app_password: str = ""
    calendar_alerts: bool = True
    alert_times: list = None
    daily_summary: bool = True
    summary_time: str = "08:00"

    def __post_init__(self):
        if self.alert_times is None:
            self.alert_times = [15, 10, 5]

class ConfigManager:
    """Manages application configuration with file persistence."""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self._load_config()
        self._setup_logging()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    return self._merge_configs(DEFAULT_CONFIG, config)
            else:
                # Create default config file
                self._save_config(DEFAULT_CONFIG)
                return DEFAULT_CONFIG
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return DEFAULT_CONFIG
    
    def _merge_configs(self, default: Dict, user: Dict) -> Dict:
        """Recursively merge user config with defaults."""
        result = default.copy()
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving config: {e}")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key."""
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by dot notation key."""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        self._save_config(self.config)
    
    def get_voice_config(self) -> VoiceConfig:
        """Get voice configuration."""
        voice_config = self.get('voice', {})
        return VoiceConfig(**voice_config)
    
    def get_ollama_config(self) -> OllamaConfig:
        """Get Ollama configuration."""
        ai_config = self.get('ai_model', {})
        return OllamaConfig(
            name=ai_config.get('name', 'llama3.1:8b'),
            server_url=ai_config.get('ollama_server', 'http://127.0.0.1:11434'),
            timeout=ai_config.get('timeout', 30)
        )
    
    def get_gmail_config(self) -> GmailConfig:
        """Get Gmail configuration."""
        gmail_config = self.get('gmail', {})
        
        # Handle both nested and flattened calendar_alerts structure
        calendar_alerts = gmail_config.get('calendar_alerts', {})
        
        # If calendar_alerts is a boolean, use default values for nested fields
        if isinstance(calendar_alerts, bool):
            calendar_alerts_enabled = calendar_alerts
            alert_times = gmail_config.get('alert_times', [15, 10, 5])
            daily_summary = gmail_config.get('daily_summary', True)
            summary_time = gmail_config.get('summary_time', '08:00')
        else:
            # Handle nested structure (backward compatibility)
            calendar_alerts_enabled = calendar_alerts.get('enabled', True)
            alert_times = calendar_alerts.get('alert_times', [15, 10, 5])
            daily_summary = calendar_alerts.get('daily_summary', True)
            summary_time = calendar_alerts.get('summary_time', '08:00')
        
        return GmailConfig(
            enabled=gmail_config.get('enabled', False),
            email=gmail_config.get('email', ''),
            app_password=gmail_config.get('app_password', ''),
            calendar_alerts=calendar_alerts_enabled,
            alert_times=alert_times,
            daily_summary=daily_summary,
            summary_time=summary_time
        )
    
    def update_voice_config(self, voice_config: VoiceConfig) -> None:
        """Update voice configuration."""
        self.set('voice', asdict(voice_config))
    
    def update_ollama_config(self, ollama_config: OllamaConfig) -> None:
        """Update Ollama configuration."""
        self.set('ai_model', asdict(ollama_config))
    
    def update_gmail_config(self, gmail_config: GmailConfig) -> None:
        """Update Gmail configuration."""
        # Convert to flattened format to match current config structure
        gmail_dict = {
            'enabled': gmail_config.enabled,
            'email': gmail_config.email,
            'app_password': gmail_config.app_password,
            'calendar_alerts': gmail_config.calendar_alerts,
            'alert_times': gmail_config.alert_times,
            'daily_summary': gmail_config.daily_summary,
            'summary_time': gmail_config.summary_time
        }
        self.set('gmail', gmail_dict)
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        self.config = DEFAULT_CONFIG.copy()
        self._save_config(self.config)

# Global configuration instance
config_manager = ConfigManager()
