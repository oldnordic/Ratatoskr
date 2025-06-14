# main.py
import sys
import logging
import threading
import time
from queue import Queue
import httpx
from bs4 import BeautifulSoup
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

# LangChain Imports
from langchain.agents import AgentExecutor, create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.tools import Tool

# Browse via HTTP client to avoid GUI threading issues

def browse_search(query: str) -> str:
    """
    Perform a DuckDuckGo HTML search via HTTP and return the first result page's text.
    """
    try:
        search_url = "https://duckduckgo.com/html/"
        params = {"q": query}
        r = httpx.get(search_url, params=params, timeout=10)
        if r.status_code != 200:
            return f"❌ Search failed with status {r.status_code}."
        soup = BeautifulSoup(r.text, "html.parser")
        link_tag = soup.select_one(".result__a")
        if not link_tag or not link_tag.get("href"):
            return "❌ No results found."
        link = link_tag["href"]
        # fetch the actual page
        r2 = httpx.get(link, timeout=10)
        if r2.status_code != 200:
            return f"❌ Failed to load result page ({r2.status_code})."
        text = BeautifulSoup(r2.text, "html.parser").get_text(separator="\n")
        # return trimmed text
        return text[:2000] + "\n\n[...]"
    except Exception as e:
        logging.error(f"Error in browse_search: {e}", exc_info=True)
        return f"❌ browse_search error: {e}"


def worker_thread(queue: Queue, user_input: str, conversation_history: list, model_name: str):
    logging.info("LangChain worker thread started.")
    try:
        llm = ChatOllama(model=model_name, temperature=0.7)
        tools = [
            Tool(
                name="Web Search",
                func=perform_web_search,
                description="Use for real-time info like news, weather, current events."
            ),
            Tool(
                name="Browse Web",
                func=browse_search,
                description="Perform a full, uncapped internet search via HTTP client."
            ),
            Tool(
                name="Long-Term Memory Search",
                func=retrieve_relevant_memories,
                description="Retrieve facts from past conversations."
            ),
            Tool(
                name="Save to Memory",
                func=add_memory,
                description="Save a fact for future reference."
            )
        ]
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
        prompt = PromptTemplate.from_template(prompt_template)
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
            max_execution_time=300
        )
        chat_history = "\n".join(f"{msg['role']}: {msg['content']}" for msg in conversation_history)
        response = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
        ai_text = response.get("output", "The agent could not determine a response.")
    except Exception as e:
        logging.error(f"Error in worker_thread: {e}", exc_info=True)
        ai_text = f"Error: {e}"
    queue.put(ai_text)


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
        self.response_timer = QTimer(self)
        self.response_timer.timeout.connect(self.check_for_response)
        self.setup_ui()
        logging.info("RatatoskrApp initialized.")

    def setup_ui(self):
        mode_group = QGroupBox("Interaction Mode")
        mode_layout = QHBoxLayout()
        self.radio_hybrid = QRadioButton("Hybrid (Text & Voice)")
        self.radio_hybrid.setChecked(True)
        self.radio_hybrid.toggled.connect(lambda: self.set_interaction_mode("hybrid"))
        self.radio_voice = QRadioButton("Voice Only")
        self.radio_voice.toggled.connect(lambda: self.set_interaction_mode("voice_only"))
        self.radio_text = QRadioButton("Text Only")
        self.radio_text.toggled.connect(lambda: self.set_interaction_mode("text_only"))
        for w in (self.radio_hybrid, self.radio_voice, self.radio_text):
            mode_layout.addWidget(w)
        mode_group.setLayout(mode_layout)
        self.main_layout.addWidget(mode_group)
        self.conversation_view = QTextEdit(readOnly=True)
        self.conversation_view.setStyleSheet("font-size: 14px;")
        self.main_layout.addWidget(self.conversation_view)
        input_layout = QHBoxLayout()
        self.input_box = QLineEdit(placeholderText="Type your message or click 'Listen'…")
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

    def set_interaction_mode(self, mode: str):
        self.current_mode = mode
        logging.info(f"Switched to {mode} mode.")
        self.update_ui_for_mode()

    def update_ui_for_mode(self):
        is_text = self.current_mode == "text_only"
        is_voice = self.current_mode == "voice_only"
        self.input_box.setEnabled(not is_voice)
        self.send_button.setEnabled(not is_voice)
        self.listen_button.setEnabled(not is_text)

    def start_listening(self):
        threading.Thread(target=self.listen_and_process, daemon=True).start()
        self.set_ui_busy(True, listening=True)

    def listen_and_process(self):
        text = listen_for_command()
        QTimer.singleShot(0, lambda: self.on_speech_recognized(text))

    def on_speech_recognized(self, text: str):
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
        self.mp_queue = Queue()
        threading.Thread(target=worker_thread,
                         args=(self.mp_queue, user_text, self.conversation_history, self.model_name),
                         daemon=True).start()
        self.response_timer.start(100)

    def check_for_response(self):
        if self.mp_queue and not self.mp_queue.empty():
            self.response_timer.stop()
            ai_text = self.mp_queue.get()
            self.handle_ai_response(ai_text)

    def handle_ai_response(self, ai_text: str):
        self.set_ui_busy(False, thinking=False)
        if ai_text.startswith("Error"):
            self.handle_task_error(ai_text)
            return
        if self.current_mode != "voice_only":
            self.conversation_view.append(f"<b>Ratatoskr:</b> {ai_text}\n")
        self.conversation_history.append({"role": "assistant", "content": ai_text})
        if self.current_mode != "text_only":
            speak(ai_text)
        if self.current_mode == "voice_only":
            QTimer.singleShot(500, self.start_listening)

    def handle_task_error(self, msg: str):
        self.set_ui_busy(False)
        self.conversation_view.append(f"<b style='color:red;'>{msg}</b>\n")

    def set_ui_busy(self, busy: bool, thinking: bool=False, listening: bool=False):
        self.update_ui_for_mode()
        self.input_box.setEnabled(not busy and self.current_mode != "voice_only")
        self.send_button.setEnabled(not busy and self.current_mode != "voice_only")
        self.listen_button.setEnabled(not busy and self.current_mode != "text_only")
        cursor = self.conversation_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
        if cursor.selectedText().endswith("Thinking..."):
            cursor.removeSelectedText()
        if busy:
            if thinking:
                self.conversation_view.append("<b>Ratatoskr:</b> Thinking...")
            elif listening:
                self.listen_button.setText("Listening...")
        else:
            self.listen_button.setText("Listen 🎙️")
            self.input_box.setFocus()

if __name__ == "__main__":
    setup_logging()
    qt_app = QApplication(sys.argv)
    app = RatatoskrApp()
    app.show()
    sys.exit(qt_app.exec())
