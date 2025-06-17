"""
Tools module for Ratatoskr AI Assistant.

This module provides various tools for file management, browser control,
LibreOffice integration, screen capture, and system automation.
"""

from .browser_tool import browse_search
from .web_search import perform_web_search
from .file_manager import file_manager
from .libreoffice_controller import libreoffice_controller
from .browser_controller import browser_controller
from .screen_controller import screen_controller

# Export all tools
__all__ = [
    'browse_search',
    'perform_web_search', 
    'file_manager',
    'libreoffice_controller',
    'browser_controller',
    'screen_controller'
]

__version__ = '1.0.0'
