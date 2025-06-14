# ratatoskr/__init__.py

from PyQt6.QtCore import QObject, pyqtSignal, QUrl
from PyQt6.QtWebEngineWidgets import QWebEngineView
from bs4 import BeautifulSoup
import httpx

class BrowserBridge(QObject):
    page_loaded = pyqtSignal(str)  # emits HTML when ready

    def __init__(self, view: QWebEngineView):
        super().__init__()
        self.view = view
        view.page().loadFinished.connect(self._on_load_finished)

    def navigate(self, url: str):
        self.view.page().load(QUrl(url))

    def _on_load_finished(self, ok: bool):
        if ok:
            self.view.page().toHtml(self._emit_html)

    def _emit_html(self, html: str):
        self.page_loaded.emit(html)
