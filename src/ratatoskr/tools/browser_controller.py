"""
Browser control and management for Ratatoskr.

This module provides browser automation capabilities including tab management,
navigation, bookmark handling, and session saving/loading.
"""

import os
import platform
import subprocess
import logging
import json
import time
from typing import List, Dict, Optional, Any
from pathlib import Path
import webbrowser

class BrowserController:
    """
    Controller for browser operations.
    
    Provides browser detection, tab management, navigation,
    and session management capabilities.
    """
    
    def __init__(self):
        self.system = platform.system().lower()
        self.browsers = self._detect_browsers()
        self.default_browser = self._get_default_browser()
        self.sessions_dir = Path.home() / ".ratatoskr" / "browser_sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"BrowserController initialized for {self.system}")
        logging.info(f"Available browsers: {list(self.browsers.keys())}")
        logging.info(f"Default browser: {self.default_browser}")
    
    def _detect_browsers(self) -> Dict[str, Dict[str, Any]]:
        """Detect available browsers on the system."""
        browsers = {}
        
        if self.system == "linux":
            # Linux browser detection
            linux_browsers = {
                "firefox": {
                    "name": "Firefox",
                    "paths": ["/usr/bin/firefox", "/usr/bin/firefox-esr"],
                    "profile_arg": "--profile",
                    "headless_arg": "--headless"
                },
                "chrome": {
                    "name": "Google Chrome",
                    "paths": ["/usr/bin/google-chrome", "/usr/bin/chromium-browser"],
                    "profile_arg": "--user-data-dir",
                    "headless_arg": "--headless"
                },
                "chromium": {
                    "name": "Chromium",
                    "paths": ["/usr/bin/chromium", "/usr/bin/chromium-browser"],
                    "profile_arg": "--user-data-dir",
                    "headless_arg": "--headless"
                },
                "edge": {
                    "name": "Microsoft Edge",
                    "paths": ["/usr/bin/microsoft-edge"],
                    "profile_arg": "--user-data-dir",
                    "headless_arg": "--headless"
                }
            }
            
            for browser_id, browser_info in linux_browsers.items():
                for path in browser_info["paths"]:
                    if os.path.exists(path):
                        browsers[browser_id] = {
                            **browser_info,
                            "path": path
                        }
                        break
        
        elif self.system == "windows":
            # Windows browser detection
            program_files = os.environ.get('PROGRAMFILES', 'C:\\Program Files')
            program_files_x86 = os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)')
            
            windows_browsers = {
                "firefox": {
                    "name": "Firefox",
                    "paths": [
                        os.path.join(program_files, "Mozilla Firefox", "firefox.exe"),
                        os.path.join(program_files_x86, "Mozilla Firefox", "firefox.exe")
                    ],
                    "profile_arg": "--profile",
                    "headless_arg": "--headless"
                },
                "chrome": {
                    "name": "Google Chrome",
                    "paths": [
                        os.path.join(program_files, "Google", "Chrome", "Application", "chrome.exe"),
                        os.path.join(program_files_x86, "Google", "Chrome", "Application", "chrome.exe")
                    ],
                    "profile_arg": "--user-data-dir",
                    "headless_arg": "--headless"
                },
                "edge": {
                    "name": "Microsoft Edge",
                    "paths": [
                        os.path.join(program_files, "Microsoft", "Edge", "Application", "msedge.exe"),
                        os.path.join(program_files_x86, "Microsoft", "Edge", "Application", "msedge.exe")
                    ],
                    "profile_arg": "--user-data-dir",
                    "headless_arg": "--headless"
                }
            }
            
            for browser_id, browser_info in windows_browsers.items():
                for path in browser_info["paths"]:
                    if os.path.exists(path):
                        browsers[browser_id] = {
                            **browser_info,
                            "path": path
                        }
                        break
        
        return browsers
    
    def _get_default_browser(self) -> Optional[str]:
        """Get the default browser."""
        try:
            # Try to get default browser from system
            if self.system == "linux":
                # Check xdg-settings
                result = subprocess.run(['xdg-settings', 'get', 'default-web-browser'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    default = result.stdout.strip()
                    # Extract browser name
                    if 'firefox' in default.lower():
                        return 'firefox'
                    elif 'chrome' in default.lower():
                        return 'chrome'
                    elif 'chromium' in default.lower():
                        return 'chromium'
                    elif 'edge' in default.lower():
                        return 'edge'
            
            # Fallback to first available browser
            if self.browsers:
                return list(self.browsers.keys())[0]
                
        except Exception as e:
            logging.debug(f"Error getting default browser: {e}")
        
        return None
    
    def get_browser_info(self) -> Dict[str, Any]:
        """Get information about available browsers."""
        return {
            "system": self.system,
            "available_browsers": self.browsers,
            "default_browser": self.default_browser,
            "sessions_dir": str(self.sessions_dir)
        }
    
    def open_url(self, url: str, browser: Optional[str] = None, 
                new_tab: bool = True, headless: bool = False) -> bool:
        """
        Open a URL in the specified browser.
        
        Args:
            url: URL to open
            browser: Browser to use (None for default)
            new_tab: Whether to open in new tab
            headless: Whether to run in headless mode
            
        Returns:
            True if successful, False otherwise
        """
        try:
            browser_id = browser or self.default_browser
            if not browser_id or browser_id not in self.browsers:
                logging.error(f"Browser not available: {browser_id}")
                return False
            
            browser_info = self.browsers[browser_id]
            cmd = [browser_info["path"]]
            
            # Add headless mode if requested
            if headless:
                cmd.append(browser_info["headless_arg"])
            
            # Add new tab argument
            if new_tab and browser_id == "firefox":
                cmd.append("--new-tab")
            elif new_tab and browser_id in ["chrome", "chromium", "edge"]:
                cmd.append("--new-window")
            
            # Add URL
            cmd.append(url)
            
            logging.info(f"Opening URL in {browser_info['name']}: {url}")
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            return True
            
        except Exception as e:
            logging.error(f"Error opening URL {url}: {e}")
            return False
    
    def close_browser(self, browser: Optional[str] = None) -> bool:
        """
        Close the specified browser.
        
        Args:
            browser: Browser to close (None for default)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            browser_id = browser or self.default_browser
            if not browser_id or browser_id not in self.browsers:
                logging.error(f"Browser not available: {browser_id}")
                return False
            
            browser_name = self.browsers[browser_id]["name"]
            
            if self.system == "linux":
                # Use pkill to close browser
                subprocess.run(['pkill', '-f', browser_name.lower()], 
                             capture_output=True, timeout=10)
            elif self.system == "windows":
                # Use taskkill to close browser
                subprocess.run(['taskkill', '/f', '/im', f"{browser_name.lower()}.exe"], 
                             capture_output=True, timeout=10)
            
            logging.info(f"Closed {browser_name}")
            return True
            
        except Exception as e:
            logging.error(f"Error closing browser {browser}: {e}")
            return False
    
    def save_session(self, session_name: str, urls: List[str]) -> bool:
        """
        Save a browser session with URLs.
        
        Args:
            session_name: Name of the session
            urls: List of URLs to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            session_file = self.sessions_dir / f"{session_name}.json"
            
            session_data = {
                "name": session_name,
                "urls": urls,
                "created": time.time(),
                "browser": self.default_browser
            }
            
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            logging.info(f"Saved browser session: {session_name}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving session {session_name}: {e}")
            return False
    
    def load_session(self, session_name: str) -> Optional[List[str]]:
        """
        Load a saved browser session.
        
        Args:
            session_name: Name of the session to load
            
        Returns:
            List of URLs or None if error
        """
        try:
            session_file = self.sessions_dir / f"{session_name}.json"
            
            if not session_file.exists():
                logging.error(f"Session not found: {session_name}")
                return None
            
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            urls = session_data.get("urls", [])
            logging.info(f"Loaded browser session: {session_name} ({len(urls)} URLs)")
            return urls
            
        except Exception as e:
            logging.error(f"Error loading session {session_name}: {e}")
            return None
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all saved browser sessions."""
        sessions = []
        
        try:
            for session_file in self.sessions_dir.glob("*.json"):
                try:
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                    
                    sessions.append({
                        "name": session_data.get("name", session_file.stem),
                        "urls": session_data.get("urls", []),
                        "created": session_data.get("created", 0),
                        "browser": session_data.get("browser", "unknown")
                    })
                except Exception as e:
                    logging.warning(f"Error reading session file {session_file}: {e}")
                    continue
            
            # Sort by creation time (newest first)
            sessions.sort(key=lambda x: x["created"], reverse=True)
            
        except Exception as e:
            logging.error(f"Error listing sessions: {e}")
        
        return sessions
    
    def delete_session(self, session_name: str) -> bool:
        """
        Delete a saved browser session.
        
        Args:
            session_name: Name of the session to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            session_file = self.sessions_dir / f"{session_name}.json"
            
            if not session_file.exists():
                logging.error(f"Session not found: {session_name}")
                return False
            
            session_file.unlink()
            logging.info(f"Deleted browser session: {session_name}")
            return True
            
        except Exception as e:
            logging.error(f"Error deleting session {session_name}: {e}")
            return False
    
    def open_session(self, session_name: str, browser: Optional[str] = None) -> bool:
        """
        Open all URLs from a saved session.
        
        Args:
            session_name: Name of the session to open
            browser: Browser to use (None for default)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            urls = self.load_session(session_name)
            if not urls:
                return False
            
            success_count = 0
            for url in urls:
                if self.open_url(url, browser, new_tab=True):
                    success_count += 1
                    time.sleep(0.5)  # Small delay between opens
            
            logging.info(f"Opened session {session_name}: {success_count}/{len(urls)} URLs")
            return success_count > 0
            
        except Exception as e:
            logging.error(f"Error opening session {session_name}: {e}")
            return False
    
    def get_bookmarks(self) -> List[Dict[str, str]]:
        """Get browser bookmarks (placeholder implementation)."""
        # This would require browser-specific implementation
        # For now, return empty list
        return []
    
    def add_bookmark(self, url: str, title: str) -> bool:
        """Add a bookmark (placeholder implementation)."""
        # This would require browser-specific implementation
        logging.info(f"Bookmark added: {title} - {url}")
        return True
    
    def is_available(self) -> bool:
        """Check if any browser is available."""
        return len(self.browsers) > 0

# Global browser controller instance
browser_controller = BrowserController() 