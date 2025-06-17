"""
Custom Voice Model Dialog for Ratatoskr AI Assistant.

This dialog allows users to:
- Add custom voice models
- Remove existing custom models
- Test custom models
- View model information
"""

import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QComboBox, QSpinBox, QTextEdit, QPushButton,
    QListWidget, QListWidgetItem, QMessageBox, QFileDialog, QDialogButtonBox
)
from PyQt6.QtCore import Qt
from voice.custom_models import custom_model_manager, VoiceModel
from voice.model_discovery import model_discovery

class CustomModelDialog(QDialog):
    """Dialog for managing custom voice models."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Voice Models")
        self.setModal(True)
        self.resize(600, 500)
        self.setup_ui()
        self.load_custom_models()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()
        
        # Instructions
        instructions_group = QGroupBox("Instructions")
        instructions_layout = QVBoxLayout()
        
        instructions_text = QTextEdit()
        instructions_text.setPlainText(model_discovery.get_general_instructions())
        instructions_text.setMaximumHeight(150)
        instructions_text.setReadOnly(True)
        instructions_layout.addWidget(instructions_text)
        
        # Download button
        download_button = QPushButton("Open Model Download Page")
        download_button.clicked.connect(self.open_download_page)
        instructions_layout.addWidget(download_button)
        
        # Model recommendations
        recommendations_button = QPushButton("Show Model Recommendations")
        recommendations_button.clicked.connect(self.show_model_recommendations)
        instructions_layout.addWidget(recommendations_button)
        
        instructions_group.setLayout(instructions_layout)
        layout.addWidget(instructions_group)
        
        # Custom models list
        models_group = QGroupBox("Custom Models")
        models_layout = QVBoxLayout()
        
        self.models_list = QListWidget()
        self.models_list.itemSelectionChanged.connect(self.on_model_selected)
        models_layout.addWidget(self.models_list)
        
        # Model actions
        actions_layout = QHBoxLayout()
        
        self.add_button = QPushButton("Add Model")
        self.add_button.clicked.connect(self.add_model)
        
        self.remove_button = QPushButton("Remove Model")
        self.remove_button.clicked.connect(self.remove_model)
        self.remove_button.setEnabled(False)
        
        self.test_button = QPushButton("Test Model")
        self.test_button.clicked.connect(self.test_model)
        self.test_button.setEnabled(False)
        
        actions_layout.addWidget(self.add_button)
        actions_layout.addWidget(self.remove_button)
        actions_layout.addWidget(self.test_button)
        actions_layout.addStretch()
        
        models_layout.addLayout(actions_layout)
        models_group.setLayout(models_layout)
        layout.addWidget(models_group)
        
        # Model details
        details_group = QGroupBox("Model Details")
        details_layout = QFormLayout()
        
        self.model_name_edit = QLineEdit()
        details_layout.addRow("Name:", self.model_name_edit)
        
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        details_layout.addRow("Path:", self.model_path_edit)
        
        self.model_gender_combo = QComboBox()
        self.model_gender_combo.addItems(["male", "female", "neutral"])
        details_layout.addRow("Gender:", self.model_gender_combo)
        
        self.model_description_edit = QLineEdit()
        details_layout.addRow("Description:", self.model_description_edit)
        
        self.model_speed_spin = QSpinBox()
        self.model_speed_spin.setRange(50, 200)
        self.model_speed_spin.setValue(100)
        self.model_speed_spin.setSuffix("%")
        details_layout.addRow("Speed:", self.model_speed_spin)
        
        self.model_temp_spin = QSpinBox()
        self.model_temp_spin.setRange(10, 100)
        self.model_temp_spin.setValue(60)
        self.model_temp_spin.setSuffix("%")
        details_layout.addRow("Temperature:", self.model_temp_spin)
        
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def load_custom_models(self):
        """Load and display custom models."""
        self.models_list.clear()
        custom_models = custom_model_manager.get_custom_models()
        
        for model in custom_models:
            item = QListWidgetItem(f"{model.name} ({model.gender})")
            item.setData(Qt.ItemDataRole.UserRole, model)
            self.models_list.addItem(item)
    
    def on_model_selected(self):
        """Handle model selection."""
        current_item = self.models_list.currentItem()
        if current_item:
            model = current_item.data(Qt.ItemDataRole.UserRole)
            self.display_model_details(model)
            self.remove_button.setEnabled(True)
            self.test_button.setEnabled(True)
        else:
            self.clear_model_details()
            self.remove_button.setEnabled(False)
            self.test_button.setEnabled(False)
    
    def display_model_details(self, model: VoiceModel):
        """Display model details in the form."""
        self.model_name_edit.setText(model.name)
        self.model_path_edit.setText(model.path)
        self.model_gender_combo.setCurrentText(model.gender)
        self.model_description_edit.setText(model.description)
        self.model_speed_spin.setValue(int(model.speed * 100))
        self.model_temp_spin.setValue(int(model.temperature * 100))
    
    def clear_model_details(self):
        """Clear model details form."""
        self.model_name_edit.clear()
        self.model_path_edit.clear()
        self.model_gender_combo.setCurrentIndex(0)
        self.model_description_edit.clear()
        self.model_speed_spin.setValue(100)
        self.model_temp_spin.setValue(60)
    
    def add_model(self):
        """Add a new custom model."""
        # Open file dialog to select model directory
        model_path = QFileDialog.getExistingDirectory(
            self, "Select Model Directory", 
            str(custom_model_manager.custom_models_dir)
        )
        
        if not model_path:
            return
        
        # Validate the model path
        if not custom_model_manager.validate_model_path(model_path):
            QMessageBox.warning(
                self, "Invalid Model", 
                "The selected directory does not appear to contain a valid TTS model.\n"
                "Please ensure it contains model files (.pth) or configuration files."
            )
            return
        
        # Get model info
        model_info = custom_model_manager.get_model_info(model_path)
        
        # Pre-fill the form
        self.model_name_edit.setText(model_info["name"])
        self.model_path_edit.setText(model_path)
        self.model_description_edit.setText(f"Custom model: {model_info['name']}")
        
        # Show a message about next steps
        QMessageBox.information(
            self, "Model Selected",
            f"Model directory selected: {model_path}\n\n"
            "Please fill in the gender and description, then click 'Test Model' to verify it works.\n"
            "The model will be saved when you close this dialog."
        )
    
    def remove_model(self):
        """Remove the selected custom model."""
        current_item = self.models_list.currentItem()
        if not current_item:
            return
        
        model = current_item.data(Qt.ItemDataRole.UserRole)
        
        reply = QMessageBox.question(
            self, "Remove Model",
            f"Are you sure you want to remove the model '{model.name}'?\n\n"
            "This will only remove it from Ratatoskr's configuration.\n"
            "The model files will remain on disk.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            if custom_model_manager.remove_custom_model(model.name):
                self.load_custom_models()
                self.clear_model_details()
                QMessageBox.information(self, "Success", f"Model '{model.name}' removed successfully.")
            else:
                QMessageBox.warning(self, "Error", f"Failed to remove model '{model.name}'.")
    
    def test_model(self):
        """Test the selected custom model."""
        selected_items = self.models_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Model Selected", "Please select a model to test.")
            return
        
        model_name = selected_items[0].text()
        custom_models = custom_model_manager.get_custom_models()
        model = next((m for m in custom_models if m.name == model_name), None)
        
        if not model:
            QMessageBox.warning(self, "Model Not Found", f"Model {model_name} not found.")
            return
        
        try:
            from voice.text_to_speech import speak
            
            test_text = f"Hello, this is a test of the {model.name} voice model. It should sound natural and clear."
            
            QMessageBox.information(self, "Model Test", 
                f"Testing model: {model.name}\n"
                f"Gender: {model.gender}\n"
                f"Path: {model.path}\n\n"
                "You should hear the voice shortly.")
            
            # Use non-blocking speak function
            speak("Test", model_path=model.path)
            
        except Exception as e:
            QMessageBox.critical(self, "Test Failed", f"Failed to test model: {e}")
    
    def show_model_recommendations(self):
        """Show model recommendations."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Model Recommendations")
        dialog.setModal(True)
        dialog.resize(700, 500)
        
        layout = QVBoxLayout()
        
        # Quick start models
        quick_start_group = QLabel("Quick Start Models (Recommended for beginners):")
        quick_start_group.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(quick_start_group)
        
        quick_models = model_discovery.get_quick_start_models()
        for model in quick_models:
            model_layout = QHBoxLayout()
            
            model_info = QLabel(f"• {model.name}\n  {model.description}\n  Size: {model.size}, Quality: {model.quality}")
            model_info.setWordWrap(True)
            model_layout.addWidget(model_info)
            
            download_btn = QPushButton("Download")
            download_btn.clicked.connect(lambda checked, m=model: self.download_model(m))
            model_layout.addWidget(download_btn)
            
            layout.addLayout(model_layout)
        
        # Male voice models
        male_group = QLabel("\nMale Voice Models:")
        male_group.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(male_group)
        
        male_models = model_discovery.get_models_by_gender("male")
        for model in male_models:
            if model not in quick_models:
                model_layout = QHBoxLayout()
                
                model_info = QLabel(f"• {model.name}\n  {model.description}\n  Size: {model.size}, Quality: {model.quality}")
                model_info.setWordWrap(True)
                model_layout.addWidget(model_info)
                
                download_btn = QPushButton("Download")
                download_btn.clicked.connect(lambda checked, m=model: self.download_model(m))
                model_layout.addWidget(download_btn)
                
                layout.addLayout(model_layout)
        
        # Female voice models
        female_group = QLabel("\nFemale Voice Models:")
        female_group.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(female_group)
        
        female_models = model_discovery.get_models_by_gender("female")
        for model in female_models:
            if model not in quick_models:
                model_layout = QHBoxLayout()
                
                model_info = QLabel(f"• {model.name}\n  {model.description}\n  Size: {model.size}, Quality: {model.quality}")
                model_info.setWordWrap(True)
                model_layout.addWidget(model_info)
                
                download_btn = QPushButton("Download")
                download_btn.clicked.connect(lambda checked, m=model: self.download_model(m))
                model_layout.addWidget(download_btn)
                
                layout.addLayout(model_layout)
        
        # Close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        dialog.exec()
    
    def download_model(self, model):
        """Open download page for a specific model."""
        if model_discovery.open_model_page(model):
            QMessageBox.information(
                self, "Download Page",
                f"Opening download page for {model.name}.\n\n"
                f"Follow the instructions:\n{model_discovery.get_download_instructions(model)}"
            )
        else:
            QMessageBox.warning(
                self, "Error",
                f"Failed to open download page for {model.name}.\n"
                f"Please visit: {model.url}"
            )
    
    def open_download_page(self):
        """Open the model download page."""
        if model_discovery.open_model_page(model_discovery.curated_models[0]):  # Use first model as default
            QMessageBox.information(
                self, "Download Page",
                "Opening the Hugging Face TTS models page in your browser.\n\n"
                "Look for models with clear gender indicators (male/female)."
            )
        else:
            QMessageBox.warning(
                self, "Error",
                "Failed to open download page.\n"
                "Please visit: https://huggingface.co/models?search=tts&sort=downloads"
            )
    
    def accept(self):
        """Save any pending changes and close the dialog."""
        # Check if there are unsaved changes
        current_item = self.models_list.currentItem()
        if current_item and self.model_name_edit.text():
            # Save the current model
            model_path = self.model_path_edit.text()
            if model_path and os.path.exists(model_path):
                success = custom_model_manager.add_custom_model(
                    model_path=model_path,
                    name=self.model_name_edit.text(),
                    gender=self.model_gender_combo.currentText(),
                    description=self.model_description_edit.text(),
                    speed=self.model_speed_spin.value() / 100.0,
                    temperature=self.model_temp_spin.value() / 100.0
                )
                
                if success:
                    self.load_custom_models()
                    QMessageBox.information(self, "Success", "Model saved successfully!")
                else:
                    QMessageBox.warning(self, "Error", "Failed to save model.")
        
        super().accept() 