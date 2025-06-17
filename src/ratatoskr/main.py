"""
Ratatoskr AI Assistant - Main Application Entry Point

This module provides the main graphical user interface for the Ratatoskr AI assistant.
It integrates text and voice interaction, LangChain agent processing, and manages
the overall application lifecycle.

Key Components:
- PyQt6-based GUI with conversation interface
- Multi-modal interaction (text/voice/hybrid)
- LangChain agent integration with Ollama
- Background processing for non-blocking UI
- Error handling and logging integration
- Persistent conversation history and memory management
- Settings dialog for voice, Ollama, and Gmail configuration
"""

import sys
import logging
import threading
from queue import Queue
from typing import List, Dict, Any, Optional

# PyQt6 imports for GUI components
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QGroupBox, QRadioButton,
    QComboBox, QLabel, QMenuBar, QMenu, QMessageBox
)
from PyQt6.QtGui import QAction, QTextCursor
from PyQt6.QtCore import QTimer, QThread, pyqtSignal

# LangChain imports for AI agent functionality
from langchain.agents import AgentExecutor, create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.tools import Tool

# Local module imports
from logging_config import setup_logging
from voice.text_to_speech import speak
from voice.speech_to_text import listen_for_command
from memory.conversation_manager import ConversationManager
from tools.web_search import perform_web_search
from tools.browser_tool import browse_search
from config import config_manager
from settings_dialog import SettingsDialog
from gmail_integration import GmailService
from tools.system_tools import get_all_system_tools

# Configuration constants
DEFAULT_WINDOW_SIZE = (800, 600)
DEFAULT_WINDOW_POSITION = (100, 100)
RESPONSE_POLL_INTERVAL = 100  # milliseconds
VOICE_RESTART_DELAY = 1000  # milliseconds
MAX_TEXT_LENGTH = 2000  # characters for web content


class SpeechRecognitionWorker(QThread):
    """Worker thread for speech recognition."""
    
    speech_recognized = pyqtSignal(str)  # Signal emitted when speech is recognized
    error_occurred = pyqtSignal(str)     # Signal emitted on error
    
    def run(self):
        """Run speech recognition in background thread."""
        try:
            logging.info("SpeechRecognitionWorker: Starting speech recognition...")
            text = listen_for_command()
            logging.info(f"SpeechRecognitionWorker: Recognition completed: '{text}'")
            self.speech_recognized.emit(text)
        except Exception as e:
            logging.error(f"SpeechRecognitionWorker: Error during recognition: {e}", exc_info=True)
            self.error_occurred.emit(str(e))


def create_agent_tools(conversation_manager: ConversationManager, app_ref=None) -> List[Tool]:
    """Create tools for the agent."""
    tools = [
        Tool(
            name="memory_search",
            description="Search conversation memory for relevant information. Input: search query",
            func=lambda query: conversation_manager.search_memory(query),
        ),
        Tool(
            name="save_to_memory",
            description="Save important information to memory. Input: content to save",
            func=lambda content: conversation_manager.save_to_memory(content),
        ),
        Tool(
            name="web_search",
            description="Search the web for current information. Input: search query",
            func=perform_web_search,
        ),
        Tool(
            name="browse_web",
            description="Browse a specific website and extract content. Input: URL",
            func=lambda url: browse_search(url, app_ref),
        ),
    ]
    
    # Add system tools
    system_tools = get_all_system_tools()
    tools.extend(system_tools)
    
    return tools


def create_agent_prompt() -> PromptTemplate:
    """
    Create the prompt template that guides the agent through the ReAct reasoning loop.
    
    Returns:
        PromptTemplate: Configured prompt for agent decision making
    """
    prompt_template = '''
You are a helpful AI assistant named Ratatoskr. Answer the user's questions as best as you can.
You have access to the following tools:
{tools}

To use a tool, use this format:

Thought: Do I need to use a tool? Yes
Action: one of [{tool_names}]
Action Input: the input to the action
Observation: the result

When done or if no tool is needed:

Thought: Do I need to use a tool? No
Final Answer: [your response]

Begin!

Previous Chat History:
{chat_history}

New Input: {input}
Thought:{agent_scratchpad}
'''
    return PromptTemplate.from_template(prompt_template)


def worker_thread(queue: Queue, user_input: str, conversation_manager: ConversationManager, 
                 ollama_config, app_ref=None) -> None:
    """
    Background thread that runs the LangChain agent for non-blocking UI.
    
    Args:
        queue: Queue for returning results to the main thread
        user_input: The user's input text
        conversation_manager: The conversation manager for context and memory
        ollama_config: Ollama configuration object
        app_ref: Optional application reference for browser-based navigation
    """
    logging.info("LangChain worker thread started.")
    try:
        # Initialize the language model with configuration
        llm = ChatOllama(
            model=ollama_config.name,
            base_url=ollama_config.server_url,
            temperature=0.7
        )
        
        # Create tools and agent components
        tools = create_agent_tools(conversation_manager, app_ref)
        prompt = create_agent_prompt()
        
        # Create the ReAct agent with reasoning capabilities
        agent = create_react_agent(llm, tools, prompt)
        
        # Configure the agent executor with safety limits
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
            max_execution_time=ollama_config.timeout
        )
        
        # Get conversation context including recent history and relevant memories
        conversation_context = conversation_manager.get_conversation_context(user_input)
        
        # Execute the agent with user input and context
        response = agent_executor.invoke({
            "input": user_input, 
            "chat_history": conversation_context
        })
        ai_text = response.get("output", "The agent could not determine a response.")
        
    except Exception as e:
        logging.error(f"Error in worker_thread: {e}", exc_info=True)
        ai_text = f"Error: {e}"
    
    # Return result to main thread
    queue.put(ai_text)


class RatatoskrApp(QMainWindow):
    """
    Main application window for the Ratatoskr AI assistant.
    
    Provides a graphical interface with conversation display, input controls,
    and mode switching between text, voice, and hybrid interaction modes.
    """
    
    def __init__(self):
        """Initialize the main application window and UI components."""
        super().__init__()
        
        # Configure window properties
        self.setWindowTitle("Ratatoskr AI Assistant (LangChain)")
        self.setGeometry(*DEFAULT_WINDOW_POSITION, *DEFAULT_WINDOW_SIZE)
        
        # Load configuration
        self._load_configuration()
        
        # Initialize conversation manager
        self.conversation_manager = ConversationManager()
        
        # Initialize speech recognition worker
        self.speech_worker = None
        
        # Set default interaction mode
        self.current_mode = "hybrid"
        
        # Initialize application state
        self.mp_queue: Optional[Queue] = None
        
        # Set app reference for browser-based navigation
        self.app_ref = self
        
        # Initialize Gmail service
        self.gmail_service = None
        if self.gmail_config.enabled:
            self.gmail_service = GmailService(config_manager)
            if self.gmail_service.authenticate():
                self.gmail_service.start_alert_monitoring(self.handle_gmail_alert)
        
        # Set up UI components
        self._setup_central_widget()
        self._setup_response_timer()
        self.setup_ui()
        
        # Display memory stats on startup
        self._display_memory_stats()
        
        logging.info("RatatoskrApp initialized successfully.")
    
    def _load_configuration(self) -> None:
        """Load application configuration from config_manager."""
        self.ollama_config = config_manager.get_ollama_config()
        self.voice_config = config_manager.get_voice_config()
        self.gmail_config = config_manager.get_gmail_config()
        self.model_name = self.ollama_config.name
        logging.info(f"Loaded configuration: model={self.model_name}, ollama={self.ollama_config}, voice={self.voice_config}, gmail={self.gmail_config}")
    
    def _setup_central_widget(self) -> None:
        """Set up the central widget and main layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)
    
    def _setup_response_timer(self) -> None:
        """Set up the timer for polling worker thread responses."""
        self.response_timer = QTimer()
        self.response_timer.timeout.connect(self.check_for_response)
    
    def setup_ui(self) -> None:
        """Set up the complete user interface."""
        self._create_menu_bar()
        self._create_mode_selector()
        self._create_conversation_view()
        self._create_input_controls()
    
    def _create_menu_bar(self) -> None:
        """Create the menu bar with settings option."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        # Settings action
        settings_action = QAction("Settings", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self.show_settings)
        file_menu.addAction(settings_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        # About action
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def _create_mode_selector(self) -> None:
        """Create the interaction mode selector."""
        mode_group = QGroupBox("Interaction Mode")
        mode_layout = QHBoxLayout()
        
        # Mode selection buttons
        self.hybrid_radio = QRadioButton("Hybrid (Text + Voice)")
        self.voice_radio = QRadioButton("Voice Only")
        self.text_radio = QRadioButton("Text Only")
        
        # Set default mode
        self.hybrid_radio.setChecked(True)
        
        # Connect signals
        self.hybrid_radio.toggled.connect(lambda: self.set_interaction_mode("hybrid"))
        self.voice_radio.toggled.connect(lambda: self.set_interaction_mode("voice_only"))
        self.text_radio.toggled.connect(lambda: self.set_interaction_mode("text_only"))
        
        mode_layout.addWidget(self.hybrid_radio)
        mode_layout.addWidget(self.voice_radio)
        mode_layout.addWidget(self.text_radio)
        mode_group.setLayout(mode_layout)
        self.main_layout.addWidget(mode_group)
    
    def _create_conversation_view(self) -> None:
        """Create the conversation display area."""
        self.conversation_view = QTextEdit(readOnly=True)
        self.conversation_view.setStyleSheet("font-size: 14px;")
        self.main_layout.addWidget(self.conversation_view)
    
    def _create_input_controls(self) -> None:
        """Create the input controls (text input, buttons)."""
        input_layout = QHBoxLayout()
        
        # Text input field
        self.input_box = QLineEdit(placeholderText="Type your message or click 'Listen'…")
        self.input_box.setStyleSheet("font-size: 14px; padding: 5px;")
        self.input_box.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_box)
        
        # Listen button for voice input
        self.listen_button = QPushButton("Listen 🎙️")
        self.listen_button.clicked.connect(self.start_listening)
        input_layout.addWidget(self.listen_button)
        
        # Send button for text input
        self.send_button = QPushButton("Send")
        self.send_button.setStyleSheet("font-size: 14px; padding: 5px;")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)
        
        self.main_layout.addLayout(input_layout)
    
    def _display_memory_stats(self) -> None:
        """Display memory statistics in the conversation view."""
        stats = self.conversation_manager.get_memory_stats()
        stats_text = f"""
<b>Memory Status:</b>
• Conversation entries: {stats['conversation_entries']}
• Long-term memories: {stats['long_term_memories']}
• Short-term memory entries: {stats['short_term_entries']}
• Conversation file size: {stats['conversation_file_size']} bytes
• Memory file size: {stats['memory_file_size']} bytes

Welcome to Ratatoskr AI Assistant! Your conversation history is being saved and will persist between sessions.
"""
        self.conversation_view.append(stats_text)
    
    def set_interaction_mode(self, mode: str) -> None:
        """Set the interaction mode and update UI accordingly."""
        self.current_mode = mode
        self.update_ui_for_mode()
        
        # Show popup message for voice modes
        if mode in ['hybrid', 'voice_only']:
            if mode == 'hybrid':
                QMessageBox.information(self, "Voice Mode Active", 
                    "Hybrid mode enabled! You can now:\n"
                    "• Type messages in the text box\n"
                    "• Click the microphone button to speak\n"
                    "• Use voice commands for system operations")
            else:  # voice_only
                QMessageBox.information(self, "Voice-Only Mode Active", 
                    "Voice-only mode enabled! You can now:\n"
                    "• Speak naturally to interact with Ratatoskr\n"
                    "• Use voice commands for all operations\n"
                    "• The app will automatically listen for your voice")
        
        logging.info(f"Interaction mode set to: {mode}")
    
    def update_ui_for_mode(self) -> None:
        """Update UI elements based on current interaction mode."""
        if self.current_mode == "text_only":
            self.input_box.setEnabled(True)
            self.send_button.setEnabled(True)
            self.listen_button.setEnabled(False)
            self.listen_button.setText("Voice Disabled")
        elif self.current_mode == "voice_only":
            self.input_box.setEnabled(False)
            self.send_button.setEnabled(False)
            self.listen_button.setEnabled(True)
            self.listen_button.setText("Listen 🎙️")
        else:  # hybrid
            self.input_box.setEnabled(True)
            self.send_button.setEnabled(True)
            self.listen_button.setEnabled(True)
            self.listen_button.setText("Listen 🎙️")
    
    def start_listening(self) -> None:
        """Begin capturing audio from the microphone using worker thread."""
        if self.speech_worker and self.speech_worker.isRunning():
            logging.info("Speech recognition already in progress, ignoring start request")
            return
            
        # Create and configure speech recognition worker
        self.speech_worker = SpeechRecognitionWorker()
        self.speech_worker.speech_recognized.connect(self.on_speech_recognized)
        self.speech_worker.error_occurred.connect(self.on_speech_error)
        
        # Start the worker
        self.speech_worker.start()
        self.set_ui_busy(True, listening=True)
        logging.info("Speech recognition worker started")
    
    def on_speech_recognized(self, text: str) -> None:
        """
        Handle transcribed speech from the microphone.
        Args:
            text: The transcribed text from speech recognition
        """
        logging.info(f"on_speech_recognized called with text: '{text}'")
        
        # Clean up the worker
        if self.speech_worker:
            self.speech_worker.quit()
            self.speech_worker.wait()
            self.speech_worker = None
        
        self.set_ui_busy(False, listening=False)
        
        if text:
            logging.info(f"Recognized speech: '{text}' - sending to agent.")
            self.input_box.setText(text)
            self.send_message()
        else:
            logging.warning("No speech recognized (empty text).")
            if self.current_mode == 'voice_only':
                self.start_listening()
    
    def on_speech_error(self, error: str) -> None:
        """
        Handle speech recognition error.
        Args:
            error: The error message from speech recognition
        """
        logging.error(f"Speech recognition error: {error}")
        
        # Clean up the worker
        if self.speech_worker:
            self.speech_worker.quit()
            self.speech_worker.wait()
            self.speech_worker = None
        
        self.set_ui_busy(False, listening=False)
        if self.current_mode == 'voice_only':
            self.start_listening()
    
    def send_message(self) -> None:
        """Send the user's text to the agent for processing."""
        user_text = self.input_box.text().strip()
        logging.info(f"send_message called with user_text: '{user_text}'")
        if not user_text:
            logging.warning("send_message called with empty user_text. Aborting.")
            return
        self.set_ui_busy(True, thinking=True)
        self.conversation_view.append(f"<b>You:</b> {user_text}")
        self.conversation_manager.add_user_message(user_text)
        self.input_box.clear()
        self.mp_queue = Queue()
        logging.info(f"Starting agent worker_thread for input: '{user_text}'")
        threading.Thread(
            target=worker_thread,
            args=(self.mp_queue, user_text, self.conversation_manager, self.ollama_config, self),
            daemon=True
        ).start()
        self.response_timer.start(RESPONSE_POLL_INTERVAL)
    
    def check_for_response(self) -> None:
        """Poll the worker thread for new AI output."""
        if self.mp_queue and not self.mp_queue.empty():
            self.response_timer.stop()
            ai_text = self.mp_queue.get()
            self.handle_ai_response(ai_text)
    
    def handle_ai_response(self, ai_text: str) -> None:
        """
        Display AI output and optionally speak it.
        Args:
            ai_text: The AI's response text
        """
        logging.info(f"handle_ai_response called with ai_text: '{ai_text}'")
        self.set_ui_busy(False, thinking=False)
        
        # Check for error responses
        if ai_text.startswith("Error") or ai_text.startswith("Failed") or "error" in ai_text.lower():
            logging.error(f"Agent returned error: {ai_text}")
            self.handle_task_error(ai_text)
            return
        
        # Check for empty or invalid responses
        if not ai_text or not ai_text.strip():
            logging.warning("AI response is empty. Nothing to speak.")
            self.conversation_view.append("<b style='color:orange;'>No response generated. Please try again.</b>\n")
            return
        
        # Display the response
        if self.current_mode != "voice_only":
            self.conversation_view.append(f"<b>Ratatoskr:</b> {ai_text}\n")
        
        # Add to conversation history
        self.conversation_manager.add_assistant_message(ai_text)
        
        # Only speak if not in text-only mode and response is not an error
        if self.current_mode != "text_only":
            logging.info(f"Calling TTS to speak: '{ai_text}'")
            speak(ai_text)
        
        # Restart listening in voice-only mode
        if self.current_mode == "voice_only":
            QTimer.singleShot(VOICE_RESTART_DELAY, self.start_listening)
    
    def handle_task_error(self, msg: str) -> None:
        """
        Display an error message in the conversation view.
        
        Args:
            msg: The error message to display
        """
        self.set_ui_busy(False)
        self.conversation_view.append(f"<b style='color:red;'>{msg}</b>\n")
    
    def set_ui_busy(self, busy: bool, thinking: bool = False, listening: bool = False) -> None:
        """
        Enable/disable controls while background work is running.
        
        Args:
            busy: Whether the UI should be in busy state
            thinking: Whether the agent is thinking (for display purposes)
            listening: Whether the app is listening for voice input
        """
        self.update_ui_for_mode()
        
        # Update input controls state
        self.input_box.setEnabled(not busy and self.current_mode != "voice_only")
        self.send_button.setEnabled(not busy and self.current_mode != "voice_only")
        self.listen_button.setEnabled(not busy and self.current_mode != "text_only")
        
        # Handle "Thinking..." message in conversation view
        cursor = self.conversation_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
        if cursor.selectedText().endswith("Thinking..."):
            cursor.removeSelectedText()
        
        # Update UI based on current state
        if busy:
            if thinking:
                self.conversation_view.append("<b>Ratatoskr:</b> Thinking...")
            elif listening:
                self.listen_button.setText("Listening...")
        else:
            self.listen_button.setText("Listen 🎙️")
            self.input_box.setFocus()
    
    def show_settings(self) -> None:
        """Show the settings dialog."""
        dialog = SettingsDialog(self)
        if dialog.exec() == SettingsDialog.DialogCode.Accepted:
            # Reload configuration after settings change
            self.ollama_config = config_manager.get_ollama_config()
            self.voice_config = config_manager.get_voice_config()
            self.gmail_config = config_manager.get_gmail_config()
            
            # Reinitialize Gmail service if needed
            if self.gmail_config.enabled and not self.gmail_service:
                self.gmail_service = GmailService(config_manager)
                if self.gmail_service.authenticate():
                    self.gmail_service.start_alert_monitoring(self.handle_gmail_alert)
            elif not self.gmail_config.enabled and self.gmail_service:
                self.gmail_service.stop_alert_monitoring()
                self.gmail_service = None
    
    def show_about(self) -> None:
        """Show the about dialog."""
        QMessageBox.about(self, "About Ratatoskr", 
            "Ratatoskr AI Assistant\n\n"
            "A powerful AI assistant with voice interaction, "
            "memory management, and Gmail integration.\n\n"
            "Version: 1.0\n"
            "Built with PyQt6 and LangChain")
    
    def handle_gmail_alert(self, message: str) -> None:
        """Handle Gmail calendar alerts."""
        self.conversation_view.append(f"<b>📅 Calendar Alert:</b> {message}\n")
        # Speak the alert if in voice mode
        if self.current_mode != "text_only":
            speak(message)
    
    def closeEvent(self, event) -> None:
        """Handle application close event."""
        if self.gmail_service:
            self.gmail_service.stop_alert_monitoring()
        event.accept()


def main() -> None:
    """Main application entry point."""
    # Set up logging
    setup_logging()
    
    # Create and run the Qt application
    qt_app = QApplication(sys.argv)
    app = RatatoskrApp()
    app.show()
    
    # Start the event loop
    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
