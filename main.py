import sys
import logging
import multiprocessing
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QGroupBox, QRadioButton
)
from PyQt6.QtGui import QTextCursor
from PyQt6.QtCore import QTimer

# Import our modules
from logging_config import setup_logging
from voice.text_to_speech import speak
from voice.speech_to_text import listen_for_command
from memory.long_term import add_memory, retrieve_relevant_memories
from tools.web_search import perform_web_search

# --- LangChain Imports ---
from langchain.agents import AgentExecutor, create_react_agent
from langchain_ollama import ChatOllama # <-- CORRECTED IMPORT
from langchain_core.prompts import PromptTemplate
from langchain.tools import Tool

# --- Top-level function for the worker process ---
def worker_process(queue, user_input, conversation_history, model_name):
    """
    This worker process uses a LangChain Agent to generate responses.
    """
    import logging
    
    try:
        logging.info("LangChain worker process started.")
        llm = ChatOllama(model=model_name, temperature=0.7)

        tools = [
            Tool(name="Web Search", func=perform_web_search, description="Use for real-time information like news, weather, or current events."),
            Tool(name="Long-Term Memory Search", func=retrieve_relevant_memories, description="Use to retrieve specific facts from past conversations, like names or user preferences."),
            Tool(name="Save to Memory", func=add_memory, description="Use to save a specific fact from the user's input for future reference.")
        ]

        # --- FIX: The corrected prompt with the required {tools} and {tool_names} variables ---
        prompt_template = """
        You are a helpful AI assistant named Ratatoskr. Answer the user's questions as best as you can.
        You have access to the following tools:
        {tools}

        To use a tool, please use the following format:

        Thought: Do I need to use a tool? Yes
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action

        When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

        Thought: Do I need to use a tool? No
        Final Answer: [your response here]

        Begin!

        Previous Chat History:
        {chat_history}

        New Input: {input}
        Thought:{agent_scratchpad}"""
        prompt = PromptTemplate.from_template(prompt_template)
        
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=6)

        chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
        
        response = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history,
        })
        
        ai_text = response.get("output", "The agent could not determine a response.")
        
        queue.put(ai_text)
    except Exception as e:
        logging.error(f"Error in LangChain worker process: {e}", exc_info=True)
        queue.put(f"Error in LangChain process: {e}")

# --- Main Application Window ---
class RatatoskrApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ratatoskr AI Assistant (LangChain)")
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
        logging.info("RatatoskrApp initialized with LangChain.")

    def setup_ui(self):
        # UI setup remains the same
        mode_group = QGroupBox("Interaction Mode")
        mode_layout = QHBoxLayout()
        self.radio_hybrid = QRadioButton("Hybrid (Text & Voice)")
        self.radio_hybrid.setChecked(True)
        self.radio_hybrid.toggled.connect(lambda: self.set_interaction_mode("hybrid"))
        self.radio_voice = QRadioButton("Voice Only")
        self.radio_voice.toggled.connect(lambda: self.set_interaction_mode("voice_only"))
        self.radio_text = QRadioButton("Text Only")
        self.radio_text.toggled.connect(lambda: self.set_interaction_mode("text_only"))
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
        if self.radio_hybrid.isChecked(): self.current_mode = "hybrid"
        elif self.radio_voice.isChecked(): self.current_mode = "voice_only"
        else: self.current_mode = "text_only"
        logging.info(f"Switched to {self.current_mode} mode.")
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
        if not user_text: return
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
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
        selected_text = cursor.selectedText().strip()
        if selected_text.endswith("Thinking..."):
            cursor.removeSelectedText()
        if is_busy:
            if thinking: self.conversation_view.append("<b>Ratatoskr:</b> Thinking...")
            elif listening: self.listen_button.setText("Listening...")
        else:
            self.listen_button.setText("Listen 🎙️")
            self.input_box.setFocus()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    setup_logging()
    app = QApplication(sys.argv)
    window = RatatoskrApp()
    window.show()
    sys.exit(app.exec())