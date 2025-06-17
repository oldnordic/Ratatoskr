# 🚀 Ratatoskr System Features

## Overview

Ratatoskr now includes comprehensive system integration capabilities that allow the AI assistant to interact with your computer's file system, applications, and hardware. These features work on both Windows and Linux systems.

## 🗂️ File Management

### Features
- **Cross-platform file operations** - Works on Windows and Linux
- **File search and navigation** - Find files by name, type, or content
- **File operations** - Read, write, copy, move, and delete files
- **Directory management** - Navigate and create directories
- **File information** - Get detailed file metadata

### Available Tools
- `list_directory` - List contents of a directory
- `search_files` - Search for files by name or type
- `read_file` - Read text file contents
- `write_file` - Write content to a file
- `copy_file` - Copy files between locations
- `delete_file` - Delete files or directories
- `navigate_directory` - Change current directory
- `create_directory` - Create new directories

### Example Commands
```
"List the files in my Documents folder"
"Search for all Python files in my home directory"
"Read the contents of my notes.txt file"
"Create a new folder called 'projects'"
"Copy my resume.pdf to the desktop"
```

## 📄 LibreOffice Integration

### Features
- **Document creation** - Create Writer, Calc, Impress, and Draw documents
- **Document conversion** - Convert between different formats (PDF, DOCX, etc.)
- **Text extraction** - Extract text from LibreOffice documents
- **Document opening** - Open documents in LibreOffice
- **Format support** - Supports ODT, DOC, DOCX, XLS, XLSX, PPT, PPTX, and more

### Available Tools
- `create_document` - Create new LibreOffice documents
- `open_document` - Open documents in LibreOffice
- `convert_document` - Convert documents to different formats
- `extract_text` - Extract text from documents
- `libreoffice_info` - Get LibreOffice status and information

### Example Commands
```
"Create a new Writer document called 'meeting_notes.odt'"
"Open my spreadsheet 'budget.xlsx'"
"Convert my presentation to PDF"
"Extract text from my document 'report.docx'"
```

## 🌐 Browser Control

### Features
- **Multi-browser support** - Firefox, Chrome, Chromium, Edge
- **URL navigation** - Open URLs in new tabs or windows
- **Session management** - Save and restore browser sessions
- **Browser control** - Close browsers and manage tabs
- **Cross-platform** - Works on Windows and Linux

### Available Tools
- `open_url` - Open URLs in the browser
- `close_browser` - Close the browser
- `save_browser_session` - Save current browser session
- `load_browser_session` - Load and open saved sessions
- `list_browser_sessions` - List all saved sessions
- `browser_info` - Get browser information

### Example Commands
```
"Open Google in a new tab"
"Save my current browser session as 'work'"
"Load my saved browser session"
"Close Firefox"
"Open GitHub and Stack Overflow"
```

## 🖥️ Screen Control

### Features
- **Screenshot capture** - Capture full screen or specific windows
- **Window management** - List, focus, and close windows
- **Application launching** - Launch applications
- **Screen information** - Get screen resolution and display info
- **Screenshot management** - List and delete screenshots

### Available Tools
- `capture_screen` - Capture full screen screenshot
- `capture_window` - Capture specific window screenshot
- `list_windows` - List all open windows
- `focus_window` - Focus a specific window
- `close_window` - Close a specific window
- `launch_application` - Launch applications
- `get_screen_resolution` - Get screen resolution
- `list_screenshots` - List saved screenshots

### Example Commands
```
"Take a screenshot of my screen"
"Capture a screenshot of my browser window"
"List all open windows"
"Focus on my text editor"
"Close the calculator window"
"Launch LibreOffice Writer"
```

## 🎤 Advanced Dictation

### Features
- **Continuous dictation** - Real-time speech-to-text
- **Auto-punctuation** - Automatic punctuation insertion
- **Formatting commands** - Voice commands for text formatting
- **Multi-language support** - Support for different languages
- **Error correction** - Built-in error handling and correction

### Available Tools
- `start_dictation` - Start continuous dictation
- `stop_dictation` - Stop dictation
- `dictation_status` - Get current dictation status
- `clear_dictation` - Clear dictation text buffer

### Voice Commands
- "new paragraph" - Start a new paragraph
- "new line" - Start a new line
- "period" - Add a period
- "comma" - Add a comma
- "question mark" - Add a question mark
- "capitalize" - Capitalize the next word
- "delete word" - Delete the last word
- "undo" - Undo the last action

### Example Usage
```
"Start dictation mode"
"Dictate: Hello world period new paragraph This is a test comma and it works exclamation mark"
"Stop dictation"
```

## 🔧 System Information

### Features
- **System status** - Get information about all system components
- **Component availability** - Check if features are available
- **Configuration info** - Get current system configuration

### Available Tools
- `file_system_info` - Get file system information
- `libreoffice_status` - Check LibreOffice availability
- `browser_status` - Check browser availability
- `screen_info` - Get screen and display information

## 🎯 Integration with AI Agent

All these features are seamlessly integrated with Ratatoskr's AI agent. You can use natural language to control your system:

### Example Interactions
```
User: "Create a new document and write a meeting agenda"
Agent: Creates a LibreOffice Writer document and writes the agenda

User: "Search for all my Python files and show me the results"
Agent: Searches your file system and displays the results

User: "Take a screenshot and save it to my desktop"
Agent: Captures a screenshot and saves it to your desktop

User: "Open my browser and go to GitHub"
Agent: Opens your browser and navigates to GitHub

User: "Start dictation mode so I can write an email"
Agent: Starts dictation mode for voice-to-text input
```

## 🔒 Security and Privacy

- **Local processing** - All operations happen on your local machine
- **No cloud dependencies** - No data sent to external servers
- **User permissions** - Respects your system's file permissions
- **Safe operations** - Confirmation prompts for destructive actions

## 🛠️ Technical Requirements

### Linux
- **File operations** - Standard Linux file system
- **LibreOffice** - LibreOffice 6.0+ installed
- **Browser** - Firefox, Chrome, or Chromium
- **Screenshots** - ImageMagick or gnome-screenshot
- **Window management** - wmctrl for window operations

### Windows
- **File operations** - Windows file system
- **LibreOffice** - LibreOffice 6.0+ installed
- **Browser** - Firefox, Chrome, or Edge
- **Screenshots** - PowerShell-based capture
- **Window management** - Windows API integration

## 🚀 Getting Started

1. **Install dependencies** - Ensure LibreOffice and browsers are installed
2. **Test features** - Use the test commands to verify functionality
3. **Start using** - Begin with simple file operations and expand
4. **Voice commands** - Practice dictation with formatting commands

## 📝 Troubleshooting

### Common Issues
- **LibreOffice not found** - Install LibreOffice or check PATH
- **Browser not detected** - Install supported browser
- **Screenshot fails** - Install ImageMagick or gnome-screenshot
- **Permission errors** - Check file and directory permissions

### Debug Commands
```
"Check system status" - Get information about all components
"Test file operations" - Verify file system access
"Test LibreOffice" - Check LibreOffice integration
"Test browser control" - Verify browser functionality
```

## 🎉 What's New

This update adds:
- ✅ Cross-platform file management
- ✅ LibreOffice document control
- ✅ Browser automation and session management
- ✅ Screen capture and window control
- ✅ Advanced voice dictation with formatting
- ✅ 35+ new system tools for the AI agent
- ✅ Seamless integration with existing features

Your Ratatoskr AI assistant is now a powerful system automation tool that can help you manage files, create documents, control your browser, capture screens, and much more - all through natural language commands!

## 🖥️ OS-Aware System Controller

### Features
- **Automatic OS and desktop environment detection** (Linux/Windows, KDE/GNOME, etc.)
- **Automatic directory creation**: All required directories (`~/ratatoskr/data`, `~/ratatoskr/logs`, etc.) are created on startup if missing
- **Adaptive system commands**: All system features (terminal, file manager, text editor, calculator, settings, app launching) use the correct app for your OS and desktop
- **Linux:**
  - Scans `.desktop` files in `/usr/share/applications`, `/usr/local/share/applications`, and `~/.local/share/applications` to build a menu of launchable apps
  - Uses the right app for your desktop (e.g., `konsole`/`dolphin` for KDE, `gnome-terminal`/`nautilus` for GNOME)
- **Windows:**
  - Uses common app locations and can prompt the user for permission to scan or to manually set `.exe` paths for custom apps
  - Adapts all system commands to Windows equivalents (e.g., `cmd`, `explorer`, `notepad`, etc.)

### Usage Examples
- "Open terminal" → launches the correct terminal for your OS/desktop
- "Open file manager" → launches `dolphin`, `nautilus`, or `explorer` as appropriate
- "Open calculator", "Open text editor", "Open settings" → always uses the right app
- "Open [any app name]" → fuzzy-matches and launches any app from your system menu

All system features (file management, LibreOffice, browser, screen, dictation, etc.) are now fully OS-adaptive and robust. 