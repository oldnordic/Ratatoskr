"""
System tools for Ratatoskr AI Assistant.

This module provides integrated tools for file management, LibreOffice control,
browser management, screen capture, and dictation that can be used by the agent.
"""

import logging
from typing import Dict, List, Any, Optional
from langchain.tools import Tool

from .file_manager import file_manager
from .libreoffice_controller import libreoffice_controller
from .browser_controller import browser_controller
from .screen_controller import screen_controller
from voice.advanced_stt import dictation_controller

def create_file_tools() -> List[Tool]:
    """Create file management tools."""
    return [
        Tool(
            name="list_directory",
            description="List contents of a directory. Input: directory path (optional, uses current directory if not provided)",
            func=lambda path: str(file_manager.list_directory(path)),
        ),
        Tool(
            name="search_files",
            description="Search for files by name. Input: search query and optional file types (e.g., '.txt,.py')",
            func=lambda query: str(file_manager.search_files(query)),
        ),
        Tool(
            name="read_file",
            description="Read contents of a text file. Input: file path",
            func=lambda path: file_manager.read_file(path) or "File not found or could not be read",
        ),
        Tool(
            name="write_file",
            description="Write content to a file. Input: file path and content separated by '|'",
            func=lambda input_str: str(file_manager.write_file(*input_str.split('|', 1))),
        ),
        Tool(
            name="copy_file",
            description="Copy a file from source to destination. Input: source_path|destination_path",
            func=lambda input_str: str(file_manager.copy_file(*input_str.split('|', 1))),
        ),
        Tool(
            name="delete_file",
            description="Delete a file or directory. Input: file path",
            func=lambda path: str(file_manager.delete_file(path)),
        ),
        Tool(
            name="navigate_directory",
            description="Navigate to a directory. Input: directory path",
            func=lambda path: str(file_manager.navigate_to(path)),
        ),
        Tool(
            name="create_directory",
            description="Create a new directory. Input: directory path",
            func=lambda path: str(file_manager.create_directory(path)),
        ),
    ]

def create_libreoffice_tools() -> List[Tool]:
    """Create LibreOffice tools."""
    return [
        Tool(
            name="create_document",
            description="Create a new LibreOffice document. Input: file_path|document_type (writer/calc/impress/draw)",
            func=lambda input_str: str(libreoffice_controller.create_document(*input_str.split('|', 1))),
        ),
        Tool(
            name="open_document",
            description="Open a document in LibreOffice. Input: file path",
            func=lambda path: str(libreoffice_controller.open_document(path)),
        ),
        Tool(
            name="convert_document",
            description="Convert a document to another format. Input: input_path|output_path|output_format",
            func=lambda input_str: str(libreoffice_controller.convert_document(*input_str.split('|', 2))),
        ),
        Tool(
            name="extract_text",
            description="Extract text from a LibreOffice document. Input: file path",
            func=lambda path: libreoffice_controller.extract_text(path) or "Could not extract text",
        ),
        Tool(
            name="libreoffice_info",
            description="Get LibreOffice information and status",
            func=lambda _: str(libreoffice_controller.get_info()),
        ),
    ]

def create_browser_tools() -> List[Tool]:
    """Create browser management tools."""
    return [
        Tool(
            name="open_url",
            description="Open a URL in the default browser. Input: URL",
            func=lambda url: str(browser_controller.open_url(url)),
        ),
        Tool(
            name="close_browser",
            description="Close the default browser",
            func=lambda _: str(browser_controller.close_browser()),
        ),
        Tool(
            name="save_browser_session",
            description="Save current browser session. Input: session_name|url1,url2,url3",
            func=lambda input_str: str(browser_controller.save_session(*input_str.split('|', 1))),
        ),
        Tool(
            name="load_browser_session",
            description="Load and open a saved browser session. Input: session_name",
            func=lambda session_name: str(browser_controller.open_session(session_name)),
        ),
        Tool(
            name="list_browser_sessions",
            description="List all saved browser sessions",
            func=lambda _: str(browser_controller.list_sessions()),
        ),
        Tool(
            name="browser_info",
            description="Get browser information and status",
            func=lambda _: str(browser_controller.get_browser_info()),
        ),
    ]

def create_screen_tools() -> List[Tool]:
    """Create screen control tools."""
    return [
        Tool(
            name="capture_screen",
            description="Capture a screenshot of the entire screen. Input: optional filename",
            func=lambda filename: screen_controller.capture_screen(filename) or "Screenshot failed",
        ),
        Tool(
            name="capture_window",
            description="Capture a screenshot of a specific window. Input: window_title|optional_filename",
            func=lambda input_str: screen_controller.capture_window(*input_str.split('|', 1)) or "Window capture failed",
        ),
        Tool(
            name="list_windows",
            description="List all open windows",
            func=lambda _: str(screen_controller.get_window_list()),
        ),
        Tool(
            name="focus_window",
            description="Focus a window by title. Input: window title",
            func=lambda title: str(screen_controller.focus_window(title)),
        ),
        Tool(
            name="close_window",
            description="Close a window by title. Input: window title",
            func=lambda title: str(screen_controller.close_window(title)),
        ),
        Tool(
            name="launch_application",
            description="Launch an application. Input: app_name|optional_args",
            func=lambda input_str: str(screen_controller.launch_application(*input_str.split('|', 1))),
        ),
        Tool(
            name="get_screen_resolution",
            description="Get screen resolution",
            func=lambda _: str(screen_controller.get_screen_resolution()),
        ),
        Tool(
            name="list_screenshots",
            description="List all saved screenshots",
            func=lambda _: str(screen_controller.list_screenshots()),
        ),
    ]

def create_dictation_tools() -> List[Tool]:
    """Create dictation tools."""
    return [
        Tool(
            name="start_dictation",
            description="Start continuous dictation mode. Input: 'start' to begin",
            func=lambda _: "Dictation started. Use 'stop_dictation' to stop.",
        ),
        Tool(
            name="stop_dictation",
            description="Stop continuous dictation mode. Input: 'stop' to end",
            func=lambda _: "Dictation stopped.",
        ),
        Tool(
            name="dictation_status",
            description="Get current dictation status",
            func=lambda _: str(dictation_controller.get_dictation_status()),
        ),
        Tool(
            name="clear_dictation",
            description="Clear current dictation text buffer",
            func=lambda _: str(dictation_controller.clear_text()),
        ),
    ]

def create_system_info_tools() -> List[Tool]:
    """Create system information tools."""
    return [
        Tool(
            name="file_system_info",
            description="Get file system information",
            func=lambda _: str(file_manager.get_system_info()),
        ),
        Tool(
            name="libreoffice_status",
            description="Check LibreOffice availability and status",
            func=lambda _: str(libreoffice_controller.is_available()),
        ),
        Tool(
            name="browser_status",
            description="Check browser availability and status",
            func=lambda _: str(browser_controller.get_browser_info()),
        ),
        Tool(
            name="screen_info",
            description="Get screen and display information",
            func=lambda _: str(screen_controller.get_system_info()),
        ),
    ]

def get_all_system_tools() -> List[Tool]:
    """Get all system tools for the agent."""
    tools = []
    
    # Add file management tools
    tools.extend(create_file_tools())
    
    # Add LibreOffice tools
    tools.extend(create_libreoffice_tools())
    
    # Add browser tools
    tools.extend(create_browser_tools())
    
    # Add screen control tools
    tools.extend(create_screen_tools())
    
    # Add dictation tools
    tools.extend(create_dictation_tools())
    
    # Add system info tools
    tools.extend(create_system_info_tools())
    
    logging.info(f"Created {len(tools)} system tools for agent")
    return tools 