"""
Settings dialog for Ratatoskr AI Assistant.

This module provides a comprehensive settings GUI for:
- Voice selection (male/female)
- Ollama server configuration
- Gmail integration settings
- Calendar alert preferences
"""

import sys
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QLabel, QLineEdit, QComboBox, QCheckBox, QSpinBox,
    QPushButton, QGroupBox, QFormLayout, QTextEdit,
    QMessageBox, QTimeEdit, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, QTime
from PyQt6.QtGui import QFont

from config import config_manager, VoiceConfig, OllamaConfig, GmailConfig
from custom_model_dialog import CustomModelDialog

class SettingsDialog(QDialog):
    """Main settings dialog for Ratatoskr configuration."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ratatoskr Settings")
        self.setModal(True)
        self.resize(600, 500)
        
        # Load current configuration
        self.voice_config = config_manager.get_voice_config()
        self.ollama_config = config_manager.get_ollama_config()
        self.gmail_config = config_manager.get_gmail_config()
        
        self.setup_ui()
        self.load_current_settings()
    
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout()
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Voice settings tab
        self.voice_tab = self.create_voice_tab()
        self.tab_widget.addTab(self.voice_tab, "Voice")
        
        # Ollama settings tab
        self.ollama_tab = self.create_ollama_tab()
        self.tab_widget.addTab(self.ollama_tab, "Ollama")
        
        # Gmail settings tab
        self.gmail_tab = self.create_gmail_tab()
        self.tab_widget.addTab(self.gmail_tab, "Gmail")
        
        # Custom models tab
        self.custom_models_tab = self.create_custom_models_tab()
        self.tab_widget.addTab(self.custom_models_tab, "Custom Models")
        
        layout.addWidget(self.tab_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_settings)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self.reset_to_defaults)
        
        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.save_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def create_voice_tab(self) -> QWidget:
        """Create the voice settings tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Voice gender selection
        gender_group = QGroupBox("Voice Gender")
        gender_layout = QFormLayout()
        
        self.voice_gender_combo = QComboBox()
        self.voice_gender_combo.addItems(["Male", "Female"])
        self.voice_gender_combo.currentIndexChanged.connect(self.update_model_info)
        gender_layout.addRow("Voice Gender:", self.voice_gender_combo)
        
        gender_group.setLayout(gender_layout)
        layout.addWidget(gender_group)
        
        # Voice model settings
        model_group = QGroupBox("Voice Model Settings")
        model_layout = QFormLayout()
        
        # Show which model will be used based on gender
        self.voice_model_info = QLabel("Model will be automatically selected based on gender")
        self.voice_model_info.setStyleSheet("color: gray; font-style: italic;")
        model_layout.addRow("TTS Model:", self.voice_model_info)
        
        # Keep the combo box but make it read-only for display purposes
        self.voice_model_combo = QComboBox()
        self.voice_model_combo.addItems([
            "tts_models/en/ljspeech/fast_pitch (Fast, Male)",
            "tts_models/en/ljspeech/tacotron2-DDC (Quality, Female)",
            "tts_models/en/vctk/vits (Quality, Multiple Speakers)"
        ])
        self.voice_model_combo.setEnabled(False)  # Make it read-only
        model_layout.addRow("Available Models:", self.voice_model_combo)
        
        self.voice_speed_spin = QSpinBox()
        self.voice_speed_spin.setRange(50, 200)
        self.voice_speed_spin.setValue(120)
        self.voice_speed_spin.setSuffix("%")
        model_layout.addRow("Speech Speed:", self.voice_speed_spin)
        
        self.voice_temp_spin = QSpinBox()
        self.voice_temp_spin.setRange(10, 100)
        self.voice_temp_spin.setValue(50)
        self.voice_temp_spin.setSuffix("%")
        model_layout.addRow("Temperature:", self.voice_temp_spin)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Test voice button
        self.test_voice_button = QPushButton("Test Voice")
        self.test_voice_button.clicked.connect(self.test_voice)
        layout.addWidget(self.test_voice_button)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_ollama_tab(self) -> QWidget:
        """Create the Ollama settings tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Model settings
        model_group = QGroupBox("AI Model Settings")
        model_layout = QFormLayout()
        
        self.model_name_edit = QLineEdit()
        model_layout.addRow("Model Name:", self.model_name_edit)
        
        self.ollama_server_edit = QLineEdit()
        model_layout.addRow("Ollama Server URL:", self.ollama_server_edit)
        
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(10, 120)
        self.timeout_spin.setValue(30)
        self.timeout_spin.setSuffix(" seconds")
        model_layout.addRow("Timeout:", self.timeout_spin)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Test connection button
        self.test_connection_button = QPushButton("Test Connection")
        self.test_connection_button.clicked.connect(self.test_ollama_connection)
        layout.addWidget(self.test_connection_button)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_gmail_tab(self) -> QWidget:
        """Create the Gmail settings tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Gmail authentication
        auth_group = QGroupBox("Gmail Authentication")
        auth_layout = QFormLayout()
        
        self.gmail_enabled_check = QCheckBox("Enable Gmail Integration")
        auth_layout.addRow(self.gmail_enabled_check)
        
        self.gmail_email_edit = QLineEdit()
        self.gmail_email_edit.setPlaceholderText("your.email@gmail.com")
        auth_layout.addRow("Gmail Address:", self.gmail_email_edit)
        
        self.gmail_password_edit = QLineEdit()
        self.gmail_password_edit.setPlaceholderText("App Password (not regular password)")
        self.gmail_password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        auth_layout.addRow("App Password:", self.gmail_password_edit)
        
        auth_group.setLayout(auth_layout)
        layout.addWidget(auth_group)
        
        # Calendar alerts
        calendar_group = QGroupBox("Calendar Alerts")
        calendar_layout = QFormLayout()
        
        self.calendar_alerts_check = QCheckBox("Enable Calendar Alerts")
        calendar_layout.addRow(self.calendar_alerts_check)
        
        self.daily_summary_check = QCheckBox("Enable Daily Summary")
        calendar_layout.addRow(self.daily_summary_check)
        
        self.summary_time_edit = QTimeEdit()
        self.summary_time_edit.setDisplayFormat("HH:mm")
        calendar_layout.addRow("Daily Summary Time:", self.summary_time_edit)
        
        # Alert times
        alert_layout = QHBoxLayout()
        alert_layout.addWidget(QLabel("Alert Times (minutes before event):"))
        
        self.alert_15_check = QCheckBox("15")
        self.alert_10_check = QCheckBox("10")
        self.alert_5_check = QCheckBox("5")
        
        alert_layout.addWidget(self.alert_15_check)
        alert_layout.addWidget(self.alert_10_check)
        alert_layout.addWidget(self.alert_5_check)
        alert_layout.addStretch()
        
        calendar_layout.addRow(alert_layout)
        calendar_group.setLayout(calendar_layout)
        layout.addWidget(calendar_group)
        
        # Test buttons
        button_layout = QHBoxLayout()
        self.test_gmail_button = QPushButton("Test Gmail Connection")
        self.test_gmail_button.clicked.connect(self.test_gmail_connection)
        
        self.get_summary_button = QPushButton("Get Daily Summary")
        self.get_summary_button.clicked.connect(self.get_daily_summary)
        
        button_layout.addWidget(self.test_gmail_button)
        button_layout.addWidget(self.get_summary_button)
        layout.addLayout(button_layout)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_custom_models_tab(self) -> QWidget:
        """Create the custom models tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Custom models management
        custom_models_group = QGroupBox("Custom Voice Models")
        custom_models_layout = QVBoxLayout()
        
        # Description
        description = QLabel(
            "Manage custom voice models for different voices and accents.\n"
            "You can download models from the Coqui TTS repository and add them here."
        )
        description.setWordWrap(True)
        custom_models_layout.addWidget(description)
        
        # Manage custom models button
        self.manage_custom_models_button = QPushButton("Manage Custom Models")
        self.manage_custom_models_button.clicked.connect(self.open_custom_models_dialog)
        custom_models_layout.addWidget(self.manage_custom_models_button)
        
        custom_models_group.setLayout(custom_models_layout)
        layout.addWidget(custom_models_group)
        
        # Current custom models info
        info_group = QGroupBox("Current Custom Models")
        info_layout = QVBoxLayout()
        
        self.custom_models_info = QLabel("No custom models installed")
        self.custom_models_info.setStyleSheet("color: gray; font-style: italic;")
        info_layout.addWidget(self.custom_models_info)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def load_current_settings(self):
        """Load current settings into the UI."""
        # Voice settings
        gender_map = {"male": 0, "female": 1}
        self.voice_gender_combo.setCurrentIndex(gender_map.get(self.voice_config.gender, 0))
        
        # Update model info based on current gender
        self.update_model_info()
        
        model_map = {
            "tts_models/en/ljspeech/fast_pitch": 0,
            "tts_models/en/ljspeech/tacotron2-DDC": 1,
            "tts_models/en/vctk/vits": 2
        }
        self.voice_model_combo.setCurrentIndex(model_map.get(self.voice_config.model, 0))
        
        self.voice_speed_spin.setValue(int(self.voice_config.speed * 100))
        self.voice_temp_spin.setValue(int(self.voice_config.temperature * 100))
        
        # Ollama settings
        self.model_name_edit.setText(self.ollama_config.name)
        self.ollama_server_edit.setText(self.ollama_config.server_url)
        self.timeout_spin.setValue(self.ollama_config.timeout)
        
        # Gmail settings
        self.gmail_enabled_check.setChecked(self.gmail_config.enabled)
        self.gmail_email_edit.setText(self.gmail_config.email)
        self.gmail_password_edit.setText(self.gmail_config.app_password)
        
        self.calendar_alerts_check.setChecked(self.gmail_config.calendar_alerts)
        self.daily_summary_check.setChecked(self.gmail_config.daily_summary)
        
        # Parse summary time
        try:
            hour, minute = map(int, self.gmail_config.summary_time.split(':'))
            self.summary_time_edit.setTime(QTime(hour, minute))
        except ValueError:
            self.summary_time_edit.setTime(QTime(8, 0))
        
        # Alert times
        self.alert_15_check.setChecked(15 in self.gmail_config.alert_times)
        self.alert_10_check.setChecked(10 in self.gmail_config.alert_times)
        self.alert_5_check.setChecked(5 in self.gmail_config.alert_times)
        
        # Update custom models info
        self.update_custom_models_info()
    
    def update_model_info(self):
        """Update the model info display based on current gender selection."""
        from voice.text_to_speech import get_voice_by_gender, get_male_speaker
        gender_map = {0: "male", 1: "female"}
        current_gender = gender_map[self.voice_gender_combo.currentIndex()]
        model_to_use = get_voice_by_gender(current_gender)
        
        if current_gender == "male" and model_to_use == "tts_models/en/vctk/vits":
            male_speaker = get_male_speaker()
            self.voice_model_info.setText(f"Will use: {model_to_use} with male speaker {male_speaker}")
        else:
            self.voice_model_info.setText(f"Will use: {model_to_use}")
    
    def save_settings(self):
        """Save all settings to configuration."""
        try:
            # Voice settings
            gender_map = {0: "male", 1: "female"}
            voice_config = VoiceConfig(
                gender=gender_map[self.voice_gender_combo.currentIndex()],
                model=self.get_selected_voice_model(),
                speed=self.voice_speed_spin.value() / 100.0,
                temperature=self.voice_temp_spin.value() / 100.0
            )
            config_manager.update_voice_config(voice_config)
            
            # Ollama settings
            ollama_config = OllamaConfig(
                name=self.model_name_edit.text(),
                server_url=self.ollama_server_edit.text(),
                timeout=self.timeout_spin.value()
            )
            config_manager.update_ollama_config(ollama_config)
            
            # Gmail settings
            alert_times = []
            if self.alert_15_check.isChecked():
                alert_times.append(15)
            if self.alert_10_check.isChecked():
                alert_times.append(10)
            if self.alert_5_check.isChecked():
                alert_times.append(5)
            
            gmail_config = GmailConfig(
                enabled=self.gmail_enabled_check.isChecked(),
                email=self.gmail_email_edit.text(),
                app_password=self.gmail_password_edit.text(),
                calendar_alerts=self.calendar_alerts_check.isChecked(),
                alert_times=alert_times,
                daily_summary=self.daily_summary_check.isChecked(),
                summary_time=self.summary_time_edit.time().toString("HH:mm")
            )
            config_manager.update_gmail_config(gmail_config)
            
            QMessageBox.information(self, "Success", "Settings saved successfully!")
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings: {e}")
    
    def get_selected_voice_model(self) -> str:
        """Get the selected voice model."""
        model_map = {
            0: "tts_models/en/ljspeech/fast_pitch",
            1: "tts_models/en/ljspeech/tacotron2-DDC",
            2: "tts_models/en/vctk/vits"
        }
        return model_map[self.voice_model_combo.currentIndex()]
    
    def test_voice(self):
        """Test the current voice settings."""
        try:
            from voice.text_to_speech import speak, get_voice_by_gender
            from config import VoiceConfig
            
            # Get the current gender selection
            gender_map = {0: "male", 1: "female"}
            current_gender = gender_map[self.voice_gender_combo.currentIndex()]
            
            # Get the model that will be used based on gender
            model_to_use = get_voice_by_gender(current_gender)
            
            test_text = f"Hello, this is a test of the {current_gender} voice settings. The voice should sound natural and clear."
            
            # Show which model is being used
            QMessageBox.information(self, "Voice Test", 
                f"Testing {current_gender} voice using model: {model_to_use}\n\n"
                "You should hear the voice change based on your gender selection.")
            
            # Use non-blocking speak function instead of speak_sync
            speak(test_text)
        except Exception as e:
            QMessageBox.warning(self, "Voice Test", f"Voice test failed: {e}")
    
    def test_ollama_connection(self):
        """Test the Ollama server connection."""
        try:
            import requests
            server_url = self.ollama_server_edit.text()
            response = requests.get(f"{server_url}/api/tags", timeout=5)
            if response.status_code == 200:
                QMessageBox.information(self, "Connection Test", "Ollama server connection successful!")
            else:
                QMessageBox.warning(self, "Connection Test", f"Server responded with status: {response.status_code}")
        except Exception as e:
            QMessageBox.critical(self, "Connection Test", f"Failed to connect to Ollama server: {e}")
    
    def test_gmail_connection(self):
        """Test the Gmail connection."""
        QMessageBox.information(self, "Gmail Test", 
            "Gmail connection test requires proper OAuth setup.\n"
            "Please ensure you have the credentials.json file in the project directory.")
    
    def get_daily_summary(self):
        """Get a sample daily summary."""
        try:
            from gmail_integration import gmail_service
            if gmail_service and gmail_service.is_authenticated:
                summary = gmail_service.get_daily_summary()
                QMessageBox.information(self, "Daily Summary", summary)
            else:
                QMessageBox.warning(self, "Daily Summary", "Gmail service not available or not authenticated.")
        except Exception as e:
            QMessageBox.warning(self, "Daily Summary", f"Failed to get summary: {e}")
    
    def reset_to_defaults(self):
        """Reset all settings to defaults."""
        reply = QMessageBox.question(self, "Reset Settings", 
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            config_manager.reset_to_defaults()
            self.load_current_settings()
            QMessageBox.information(self, "Reset Complete", "Settings have been reset to defaults.")
    
    def open_custom_models_dialog(self):
        """Open the custom models management dialog."""
        dialog = CustomModelDialog(self)
        if dialog.exec() == CustomModelDialog.DialogCode.Accepted:
            # Refresh the custom models info
            self.update_custom_models_info()
    
    def update_custom_models_info(self):
        """Update the custom models info display."""
        from voice.custom_models import custom_model_manager
        custom_models = custom_model_manager.get_custom_models()
        
        if custom_models:
            info_text = f"Installed custom models ({len(custom_models)}):\n"
            for model in custom_models:
                info_text += f"• {model.name} ({model.gender}) - {model.description}\n"
        else:
            info_text = "No custom models installed"
        
        self.custom_models_info.setText(info_text) 