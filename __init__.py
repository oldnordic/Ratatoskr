"""Package utilities for the Ratatoskr assistant."""

from typing import Any

try:
    from PyQt6.QtCore import QObject, pyqtSignal, QUrl
    from PyQt6.QtWebEngineWidgets import QWebEngineView
except Exception:  # pragma: no cover - optional dependency
    class QObject:  # type: ignore[misc]
        pass

    def pyqtSignal(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[misc]
        return None

    class QUrl:  # type: ignore[misc]
        def __init__(self, _url: str) -> None:
            pass

    class QWebEngineView:  # type: ignore[misc]
        def page(self) -> Any:
            raise NotImplementedError


class BrowserBridge(QObject):
    """Bridge between a ``QWebEngineView`` and Python."""

    page_loaded = pyqtSignal(str)

    def __init__(self, view: QWebEngineView) -> None:
        super().__init__()
        self.view = view
        view.page().loadFinished.connect(self._on_load_finished)

    def navigate(self, url: str) -> None:
        self.view.page().load(QUrl(url))

    def _on_load_finished(self, ok: bool) -> None:
        if ok:
            self.view.page().toHtml(self._emit_html)

    def _emit_html(self, html: str) -> None:
        self.page_loaded.emit(html)
