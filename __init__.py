"""Package utilities for the Ratatoskr assistant.

This module exposes a small helper class used to bridge an embedded
``QWebEngineView`` with Python code.  The PyQt6 imports are optional so the
project can still run in environments where the GUI dependencies are missing.
"""

from typing import Any

# Import Qt classes if available.  When running in headless or CI environments
# these imports may fail, so fallbacks are provided below.
try:
    from PyQt6.QtCore import QObject, pyqtSignal, QUrl
    from PyQt6.QtWebEngineWidgets import QWebEngineView
except Exception:  # pragma: no cover - optional dependency
    # Dummy classes used when PyQt6 is not installed.  They provide just enough
    # of the interface for type checking and tests to run.
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
        # Notify Python once the web page has finished loading so the HTML can
        # be inspected.
        view.page().loadFinished.connect(self._on_load_finished)

    def navigate(self, url: str) -> None:
        """Load ``url`` in the associated browser view."""
        self.view.page().load(QUrl(url))

    def _on_load_finished(self, ok: bool) -> None:
        """Emit the page HTML once loading completes successfully."""
        if ok:
            self.view.page().toHtml(self._emit_html)

    def _emit_html(self, html: str) -> None:
        """Proxy method used to emit the ``page_loaded`` signal."""
        self.page_loaded.emit(html)
