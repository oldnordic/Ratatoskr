import sys
import logging
import multiprocessing
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QGroupBox, QRadioButton
)
from PyQt5.QtCore import QTimer

# Import our modules
from logging_config import setup_logging
from voice.text_to_speech import speak
from voice.speech_to_text import listen_for_command
from memory.long_term import store_memory
from tools.web_search import perform_web_search

# --- Top-level function for the worker process ---
# This must be a top-level function to be used with multiprocessing.
def worker_process(queue, user_text, conversation_history, model_name):
    """
    This function runs in a separate process and handles all the heavy lifting:
    deciding to search the web, retrieving memories, and calling the LLM.
    """
    # Imports must be inside the worker for the 'spawn' method
    import requests
    import json
    import logging
    from memory.long_term import retrieve_relevant_memories

    try:
        # --- Agentic Step 1: Decide if a web search is needed ---
        search_decision_prompt = f"""You must decide if a web search is necessary to answer the user's query. Answer only with the single word "yes" or "no".
        The user's query is: "{user_text}"
        
        Is a real-time web search required to answer this? For example, for current events, live data (like weather or stock prices), or information about recent topics.
        """
        search_decision_messages = [{'role': 'user', 'content': search_decision_prompt}]
        url = "http://127.0.0.1:11434/api/chat"
        headers = {"Content-Type": "application/json"}
        search_payload = {"model": model_name, "messages": search_decision_messages, "stream": False, "options": {"temperature": 0.0}}
        
        search_decision_response = requests.post(url, headers=headers, data=json.dumps(search_payload), timeout=20)
        search_decision_response.raise_for_status()
        search_answer = search_decision_response.json().get('message', {}).get('content', 'no').lower().strip()

        context_from_tool = ""
        if "yes" in search_answer:
            logging.info("Agent decided a web search is needed.")
            context_from_tool = perform_web_search(user_text)
        else:
            logging.info("Agent decided to use internal memory.")
            relevant_memories = retrieve_relevant_memories(user_text)
            if relevant_memories:
                context_from_tool = "Use the following retrieved context to answer the user's question: " + "; ".join(relevant_memories)

        # --- Agentic Step 2: Generate final response ---
        final_context = conversation_history.copy()
        if context_from_tool:
            final_context.insert(-1, {"role": "system", "content": context_from_tool})
        
        logging.info(f"Final context being sent to Ollama: {final_context}")
        final_payload = { "model": model_name, "messages": final_context, "stream": False }
        final_response = requests.post(url, headers=headers, data=json.dumps(final_payload), timeout=60)
        final_response.raise_for_status()
        ai_text = final_response.json().get('message', {}).get('content', '')
        
        # Store memory of the user's prompt
        store_memory(user_text)
        
        queue.put(ai_text)

    except Exception as e:
        logging.error(f"Error in worker process: {e}", exc_info=True)
        queue.put(f"Error in background process: {e}")


# --- Main Application Window ---
class RatatoskrApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ratatoskr AI Assistant")
        self.setGeometry(100, 100, 800, 600)

        try:
            from config import MODEL_NAME
            self.model_name = MODEL_NAME
        except (ImportError, AttributeError):
            self.model_name = "llama3.1:8b"

        self.conversation_history = []
        self.current_mode = "hybrid"
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        self.mp_queue = None
        self.response_timer = QTimer(self)
        self.response_timer.timeout.connect(self.check_for_response)
        
        self.setup_ui()
        logging.info("RatatoskrApp initialized.")

    def setup_ui(self):
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
        from threading import Thread
        self.set_ui_busy(True, listening=True)
        thread = Thread(target=self.listen_and_process)
        thread.daemon = True
        thread.start()

    def listen_and_process(self):
        text = listen_for_command()
        QTimer.singleShot(0, lambda: self.on_speech_recognized(text))

    def on_speech_recognized(self, text):
        self.set_ui_busy(False, listening=False)
        if text:
            self.input_box.setText(text)
            self.send_message()
        elif self.current_mode == 'voice_only':
            self.start_listening()

    def send_message(self):
        user_text = self.input_box.text().strip()
        if not user_text:
            return
        
        self.set_ui_busy(True, thinking=True)
        self.conversation_view.append(f"<b>You:</b> {user_text}")
        self.conversation_history.append({"role": "user", "content": user_text})
        self.input_box.clear()

        from multiprocessing import Process, Queue
        self.mp_queue = Queue()
        process_args = (self.mp_queue, user_text, self.conversation_history, self.model_name)
        self.ai_process = Process(target=worker_process, args=process_args)
        self.ai_process.daemon = True
        self.ai_process.start()
        self.response_timer.start(100)
    
    def check_for_response(self):
        if self.mp_queue and not self.mp_queue.empty():
            self.response_timer.stop()
            ai_text = self.mp_queue.get()
            self.handle_ai_response(ai_text)

    def handle_ai_response(self, ai_text):
        self.set_ui_busy(False, thinking=False)
        
        if ai_text.startswith("Error:"):
             self.handle_task_error(ai_text)
             return

        if self.current_mode != "voice_only":
            self.conversation_view.append(f"<b>Ratatoskr:</b> {ai_text}\n")
        self.conversation_history.append({"role": "assistant", "content": ai_text})
        if self.current_mode != "text_only":
            speak(ai_text)
        if self.current_mode == "voice_only":
            QTimer.singleShot(500, self.start_listening)
        
    def handle_task_error(self, error_message):
        self.set_ui_busy(False, thinking=False, listening=False)
        self.conversation_view.append(f"<b style='color:red;'>{error_message}</b>\n")
        
    def set_ui_busy(self, is_busy, thinking=True, listening=False):
        self.update_ui_for_mode()
        if is_busy:
            self.input_box.setEnabled(False)
            self.send_button.setEnabled(False)
            self.listen_button.setEnabled(False)

        cursor = self.conversation_view.textCursor()
        cursor.movePosition(cursor.End)
        cursor.select(cursor.BlockUnderCursor)
        selected_text = cursor.selectedText().strip()
        if selected_text.endswith("Thinking..."):
            cursor.removeSelectedText()
        if is_busy:
            if thinking:
                self.conversation_view.append("<b>Ratatoskr:</b> Thinking...")
            elif listening:
                self.listen_button.setText("Listening...")
        else:
            self.listen_button.setText("Listen 🎙️")
            self.input_box.setFocus()

if __name__ == "__main__":
    # Set the start method to 'spawn' to avoid CUDA/forking conflicts
    multiprocessing.set_start_method('spawn', force=True)
    
    setup_logging()
    
    app = QApplication(sys.argv)
    window = RatatoskrApp()
    window.show()
    sys.exit(app.exec_())