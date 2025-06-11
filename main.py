import sys
import logging
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QGroupBox, QRadioButton
)
from PyQt5.QtCore import QObject, QThread, pyqtSignal

# Import our modules
from logging_config import setup_logging
from llm.ollama_client import get_ai_response
from voice.text_to_speech import speak
from voice.speech_to_text import listen_for_command
from memory.long_term import retrieve_relevant_memories, store_memory

# --- Worker for background tasks ---
class Worker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, task_func, *args, **kwargs):
        super().__init__()
        self.task_func = task_func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            logging.info(f"Starting background task: {self.task_func.__name__}")
            result = self.task_func(*self.args, **self.kwargs)
            logging.info(f"Background task {self.task_func.__name__} finished successfully.")
            self.finished.emit(result)
        except Exception as e:
            import traceback
            logging.error(f"Error in background task: {traceback.format_exc()}")
            self.error.emit(str(e))

# --- Main Application Window ---
class MycroftApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mycroft AI Assistant")
        self.setGeometry(100, 100, 800, 600)

        self.conversation_history = []
        self.running_threads = []
        self.current_mode = "hybrid"
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.setup_ui()
        logging.info("MycroftApp initialized.")

    def setup_ui(self):
        # --- Mode Selection ---
        mode_group = QGroupBox("Interaction Mode")
        mode_layout = QHBoxLayout()
        self.radio_hybrid = QRadioButton("Hybrid (Text & Voice)")
        self.radio_hybrid.setChecked(True)
        self.radio_hybrid.toggled.connect(lambda checked: self.set_interaction_mode("hybrid") if checked else None)
        self.radio_voice = QRadioButton("Voice Only")
        self.radio_voice.toggled.connect(lambda checked: self.set_interaction_mode("voice_only") if checked else None)
        self.radio_text = QRadioButton("Text Only")
        self.radio_text.toggled.connect(lambda checked: self.set_interaction_mode("text_only") if checked else None)
        mode_layout.addWidget(self.radio_hybrid)
        mode_layout.addWidget(self.radio_voice)
        mode_layout.addWidget(self.radio_text)
        mode_group.setLayout(mode_layout)
        self.main_layout.addWidget(mode_group)

        # --- Other UI Components ---
        self.conversation_view = QTextEdit()
        self.conversation_view.setReadOnly(True)
        self.conversation_view.setStyleSheet("font-size: 14px;")
        self.main_layout.addWidget(self.conversation_view)
        input_layout = QHBoxLayout()
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Type your message or click 'Listen'...")
        self.input_box.setStyleSheet("font-size: 14px; padding: 5px;")
        self.input_box.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_box)
        self.listen_button = QPushButton("Listen 🎙️")
        self.listen_button.clicked.connect(self.start_listening)
        input_layout.addWidget(self.listen_button)
        self.send_button = QPushButton("Send")
        self.send_button.setStyleSheet("font-size: 14px; padding: 5px;")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)
        self.main_layout.addLayout(input_layout)
        
    def set_interaction_mode(self, mode):
        self.current_mode = mode
        logging.info(f"Switched to {mode} mode.")
        self.update_ui_for_mode()

    def update_ui_for_mode(self):
        is_text_mode = self.current_mode == 'text_only'
        is_voice_mode = self.current_mode == 'voice_only'
        self.input_box.setEnabled(not is_voice_mode)
        self.send_button.setEnabled(not is_voice_mode)
        self.listen_button.setEnabled(not is_text_mode)

    def start_listening(self):
        self.set_ui_busy(True, listening=True)
        self.run_in_thread(listen_for_command, self.on_speech_recognized)

    def on_speech_recognized(self, text):
        self.set_ui_busy(False, listening=False)
        if text:
            self.input_box.setText(text)
            self.send_message()

    def send_message(self):
        user_text = self.input_box.text().strip()
        if not user_text:
            return
        self.set_ui_busy(True)
        self.conversation_view.append(f"<b>You:</b> {user_text}")
        self.conversation_history.append({"role": "user", "content": user_text})
        self.input_box.clear()
        self.run_in_thread(self.prepare_and_get_response, self.handle_ai_response, user_text=user_text)

    def prepare_and_get_response(self, user_text):
        """This function runs in the background to prepare context and get an AI response."""
        logging.info("Preparing response...")
        
        logging.info("Step 1: Retrieving relevant memories...")
        relevant_memories = retrieve_relevant_memories(user_text)
        logging.info("Step 1 complete.")
        
        logging.info("Step 2: Assembling context for LLM...")
        current_context = self.conversation_history.copy()
        if relevant_memories:
            memory_context = "Remember these potentially relevant facts from past conversations: " + "; ".join(relevant_memories)
            # Insert system prompt with memories before the last user message
            current_context.insert(-1, {"role": "system", "content": memory_context})
        logging.info("Step 2 complete.")
        
        logging.info("Step 3: Calling LLM...")
        ai_text = get_ai_response(history=current_context)
        logging.info("Step 3 complete.")
        
        return ai_text

    def handle_ai_response(self, ai_text):
        self.set_ui_busy(False, thinking=False)
        if self.current_mode != "voice_only":
            self.conversation_view.append(f"<b>Mycroft:</b> {ai_text}\n")
        self.conversation_history.append({"role": "assistant", "content": ai_text})
        if self.current_mode != "text_only":
            speak(ai_text)
        last_user_message = self.conversation_history[-2]['content']
        store_memory(last_user_message)

    def handle_task_error(self, error_message):
        logging.error(f"Displaying error in GUI: {error_message}")
        self.set_ui_busy(False, thinking=False, listening=False)
        self.conversation_view.append(f"<b style='color:red;'>Error:</b> {error_message}\n")

    def run_in_thread(self, task_func, on_finish_slot, *args, **kwargs):
        thread = QThread()
        worker = Worker(task_func, *args, **kwargs)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(on_finish_slot)
        worker.error.connect(self.handle_task_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self.running_threads.append(thread)
        thread.finished.connect(lambda: self.running_threads.remove(thread))
        thread.start()
        
    def set_ui_busy(self, is_busy, thinking=True, listening=False):
        self.update_ui_for_mode() # First, set buttons to their default state for the current mode
        # Then, disable all if busy
        if is_busy:
            self.input_box.setEnabled(False)
            self.send_button.setEnabled(False)
            self.listen_button.setEnabled(False)

        # Status message handling
        cursor = self.conversation_view.textCursor()
        cursor.movePosition(cursor.End)
        cursor.select(cursor.BlockUnderCursor)
        selected_text = cursor.selectedText().strip()
        if selected_text.endswith("Thinking...") or selected_text.endswith("Listening..."):
            cursor.removeSelectedText()
        if is_busy:
            if thinking:
                self.conversation_view.append("<b>Mycroft:</b> Thinking...")
            elif listening:
                self.listen_button.setText("Listening...")
        else: # Not busy, so restore default listen button text
            self.listen_button.setText("Listen 🎙️")
            self.input_box.setFocus()

if __name__ == "__main__":
    setup_logging()
    app = QApplication(sys.argv)
    window = MycroftApp()
    window.show()
    sys.exit(app.exec_())