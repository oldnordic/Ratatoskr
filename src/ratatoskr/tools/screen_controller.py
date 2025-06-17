"""
Screen control and window management for Ratatoskr.

This module provides screen capture, window management, and application
control capabilities across different operating systems.
"""

import os
import platform
import subprocess
import logging
import time
import glob
import configparser
import shlex
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from difflib import get_close_matches

class ScreenController:
    """
    Controller for screen operations and window management.
    
    Provides screen capture, window management, and application
    control capabilities.
    """
    
    def __init__(self):
        self.system = platform.system().lower()
        self.os_version = platform.version()
        self.arch = platform.machine()
        self.home_dir = Path.home()
        
        # Create necessary directories on startup
        self._create_directories()
        
        # Initialize OS-specific configurations
        self._initialize_os_specific()
        
        logging.info(f"ScreenController initialized for {self.system} {self.os_version}")
    
    def _create_directories(self):
        """Create necessary directories for Ratatoskr."""
        dirs_to_create = [
            self.home_dir / "ratatoskr",
            self.home_dir / "ratatoskr" / "data",
            self.home_dir / "ratatoskr" / "logs",
            self.home_dir / "ratatoskr" / "screenshots",
            self.home_dir / "ratatoskr" / "documents",
            self.home_dir / "ratatoskr" / "temp"
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created directory: {dir_path}")
    
    def _initialize_os_specific(self):
        """Initialize OS-specific features and configurations."""
        if self.system == "windows":
            self._initialize_windows()
        else:
            self._initialize_linux()
    
    def _initialize_windows(self):
        """Initialize Windows-specific features."""
        logging.info("Initializing Windows-specific features")
        
        # Windows-specific configurations
        self.windows_config = {
            "terminal": "cmd",
            "file_manager": "explorer",
            "text_editor": "notepad",
            "calculator": "calc",
            "settings": "ms-settings:",
            "app_dirs": [
                os.environ.get('PROGRAMFILES', 'C:\\Program Files'),
                os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)'),
                os.environ.get('LOCALAPPDATA', 'C:\\Users\\%USERNAME%\\AppData\\Local'),
                os.environ.get('APPDATA', 'C:\\Users\\%USERNAME%\\AppData\\Roaming')
            ]
        }
        
        # Check for common Windows applications
        self.windows_apps = self._scan_windows_apps()
        logging.info(f"Found {len(self.windows_apps)} Windows applications")
    
    def _initialize_linux(self):
        """Initialize Linux-specific features."""
        logging.info("Initializing Linux-specific features")
        
        # Detect desktop environment
        self.desktop_env = self._detect_desktop_environment()
        
        # Linux-specific configurations
        self.linux_config = {
            "terminal": "konsole" if self.desktop_env == "kde" else "gnome-terminal",
            "file_manager": "dolphin" if self.desktop_env == "kde" else "nautilus",
            "text_editor": "kate" if self.desktop_env == "kde" else "gedit",
            "calculator": "kcalc" if self.desktop_env == "kde" else "gnome-calculator",
            "settings": "systemsettings" if self.desktop_env == "kde" else "gnome-control-center",
            "app_dirs": [
                "/usr/bin",
                "/usr/local/bin", 
                "/opt",
                "/snap/bin",
                "/usr/share/applications",
                "/usr/local/share/applications",
                str(self.home_dir / ".local/share/applications")
            ]
        }
        
        # Scan Linux applications
        self.linux_apps = self._scan_linux_apps()
        logging.info(f"Found {len(self.linux_apps)} Linux applications")
        
        # Check for common Linux tools
        tools_to_check = ['libreoffice', 'firefox', 'vlc', 'gimp']
        for tool in tools_to_check:
            if self._find_linux_executable(tool):
                logging.info(f"Found Linux tool: {tool}")
    
    def _detect_desktop_environment(self) -> str:
        """Detect the current desktop environment."""
        try:
            # Check environment variables
            kde_vars = ['KDE_FULL_SESSION', 'KDE_SESSION_VERSION', 'XDG_CURRENT_DESKTOP']
            if any(os.environ.get(var) for var in kde_vars):
                return "kde"
            
            # Check for KDE applications
            if any(Path(p).exists() for p in ['/usr/bin/dolphin', '/usr/bin/konsole']):
                return "kde"
            
            # Check for GNOME
            if os.environ.get('XDG_CURRENT_DESKTOP', '').lower() in ['gnome', 'ubuntu:gnome']:
                return "gnome"
            
            # Default to GNOME if unsure
            return "gnome"
        except:
            return "gnome"
    
    def _find_linux_executable(self, name: str) -> Optional[str]:
        """Find Linux executable in PATH."""
        try:
            result = subprocess.run(['which', name], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return None
    
    def _scan_windows_apps(self) -> Dict[str, str]:
        """Scan Windows applications."""
        apps = {}
        
        # Common Windows applications
        common_apps = {
            "Notepad": "notepad.exe",
            "Calculator": "calc.exe",
            "Command Prompt": "cmd.exe",
            "PowerShell": "powershell.exe",
            "File Explorer": "explorer.exe",
            "Paint": "mspaint.exe",
            "WordPad": "wordpad.exe",
            "Internet Explorer": "iexplore.exe",
            "Windows Media Player": "wmplayer.exe"
        }
        
        # Add common applications
        for name, exe in common_apps.items():
            if self._find_windows_exe(exe):
                apps[name] = exe
        
        return apps
    
    def _find_windows_exe(self, exe_name: str) -> Optional[str]:
        """Find Windows executable in common locations."""
        search_paths = [
            os.environ.get('WINDIR', 'C:\\Windows'),
            os.environ.get('SYSTEMROOT', 'C:\\Windows\\System32'),
            os.environ.get('PROGRAMFILES', 'C:\\Program Files'),
            os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)')
        ]
        
        for path in search_paths:
            exe_path = os.path.join(path, exe_name)
            if os.path.exists(exe_path):
                return exe_path
        
        return None
    
    def _scan_linux_apps(self) -> Dict[str, str]:
        """Scan Linux applications from .desktop files."""
        apps = {}
        app_dirs = [
            '/usr/share/applications/',
            '/usr/local/share/applications/',
            str(self.home_dir / '.local/share/applications/')
        ]
        
        for app_dir in app_dirs:
            if os.path.exists(app_dir):
                for desktop_file in glob.glob(os.path.join(app_dir, '*.desktop')):
                    try:
                        config = configparser.ConfigParser(interpolation=None)
                        config.read(desktop_file)
                        if 'Desktop Entry' in config:
                            entry = config['Desktop Entry']
                            name = entry.get('Name')
                            exec_cmd = entry.get('Exec')
                            if name and exec_cmd and not entry.get('NoDisplay', 'false').lower() == 'true':
                                exec_cmd = shlex.split(exec_cmd)[0]
                                apps[name] = exec_cmd
                    except Exception as e:
                        logging.debug(f"Error parsing {desktop_file}: {e}")
        
        return apps
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        info = {
            "os_type": self.system,
            "os_version": self.os_version,
            "architecture": self.arch,
            "home_directory": str(self.home_dir),
            "available_apps_count": 0
        }
        
        if self.system == "windows":
            info.update({
                "config": self.windows_config,
                "available_apps_count": len(self.windows_apps)
            })
        else:
            info.update({
                "desktop_environment": self.desktop_env,
                "config": self.linux_config,
                "available_apps_count": len(self.linux_apps)
            })
        
        return info
    
    def capture_screen(self, filename: Optional[str] = None) -> Optional[str]:
        """
        Capture a screenshot of the entire screen.
        
        Args:
            filename: Name for the screenshot file (None for auto-generated)
            
        Returns:
            Path to the screenshot file or None if error
        """
        try:
            if not filename:
                timestamp = int(time.time())
                filename = f"screenshot_{timestamp}.png"
            
            screenshot_path = self.home_dir / "ratatoskr" / "screenshots" / filename
            
            if self.system == "linux":
                # Use import command (ImageMagick)
                result = subprocess.run(['import', '-window', 'root', str(screenshot_path)], 
                                      capture_output=True, timeout=10)
                if result.returncode != 0:
                    # Try alternative: gnome-screenshot
                    result = subprocess.run(['gnome-screenshot', '-f', str(screenshot_path)], 
                                          capture_output=True, timeout=10)
                    if result.returncode != 0:
                        # Try alternative: xdg-desktop-portal
                        result = subprocess.run(['xdg-desktop-portal', 'screenshot', str(screenshot_path)], 
                                              capture_output=True, timeout=10)
            
            elif self.system == "windows":
                # Use PowerShell to capture screen
                ps_script = f"""
                Add-Type -AssemblyName System.Windows.Forms
                Add-Type -AssemblyName System.Drawing
                $screen = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
                $bitmap = New-Object System.Drawing.Bitmap $screen.Width, $screen.Height
                $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
                $graphics.CopyFromScreen($screen.Left, $screen.Top, 0, 0, $screen.Size)
                $bitmap.Save('{screenshot_path}')
                $graphics.Dispose()
                $bitmap.Dispose()
                """
                
                result = subprocess.run(['powershell', '-Command', ps_script], 
                                      capture_output=True, timeout=10)
            
            if result.returncode == 0 and screenshot_path.exists():
                logging.info(f"Screenshot saved: {screenshot_path}")
                return str(screenshot_path)
            else:
                logging.error(f"Error capturing screenshot: {result.stderr}")
                return None
                
        except Exception as e:
            logging.error(f"Error capturing screen: {e}")
            return None
    
    def capture_window(self, window_title: str, filename: Optional[str] = None) -> Optional[str]:
        """
        Capture a screenshot of a specific window.
        
        Args:
            window_title: Title of the window to capture
            filename: Name for the screenshot file (None for auto-generated)
            
        Returns:
            Path to the screenshot file or None if error
        """
        try:
            if not filename:
                timestamp = int(time.time())
                safe_title = "".join(c for c in window_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                filename = f"window_{safe_title}_{timestamp}.png"
            
            screenshot_path = self.home_dir / "ratatoskr" / "screenshots" / filename
            
            if self.system == "linux":
                # Use import command with window selection
                result = subprocess.run(['import', str(screenshot_path)], 
                                      capture_output=True, timeout=30)
                # Note: This will prompt user to select window
                # For automation, we'd need a different approach
            
            elif self.system == "windows":
                # Windows implementation would require more complex window detection
                logging.warning("Window capture not fully implemented for Windows")
                return self.capture_screen(filename)
            
            if result.returncode == 0 and screenshot_path.exists():
                logging.info(f"Window screenshot saved: {screenshot_path}")
                return str(screenshot_path)
            else:
                logging.error(f"Error capturing window: {result.stderr}")
                return None
                
        except Exception as e:
            logging.error(f"Error capturing window: {e}")
            return None
    
    def get_window_list(self) -> List[Dict[str, Any]]:
        """
        Get list of open windows.
        
        Returns:
            List of window information dictionaries
        """
        windows = []
        
        try:
            if self.system == "linux":
                # Use wmctrl to get window list
                result = subprocess.run(['wmctrl', '-l'], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line:
                            parts = line.split(None, 3)
                            if len(parts) >= 4:
                                windows.append({
                                    "id": parts[0],
                                    "desktop": parts[1],
                                    "geometry": parts[2],
                                    "title": parts[3]
                                })
            
            elif self.system == "windows":
                # Use PowerShell to get window list
                ps_script = """
                Add-Type -AssemblyName System.Windows.Forms
                $windows = @()
                [System.Windows.Forms.Application]::OpenForms | ForEach-Object {
                    $windows += @{
                        'title' = $_.Text
                        'handle' = $_.Handle
                        'visible' = $_.Visible
                    }
                }
                $windows | ConvertTo-Json
                """
                
                result = subprocess.run(['powershell', '-Command', ps_script], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    import json
                    windows = json.loads(result.stdout)
            
        except Exception as e:
            logging.error(f"Error getting window list: {e}")
        
        return windows
    
    def focus_window(self, window_title: str) -> bool:
        """
        Focus a window by title.
        
        Args:
            window_title: Title of the window to focus
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.system == "linux":
                # Use wmctrl to focus window
                result = subprocess.run(['wmctrl', '-a', window_title], 
                                      capture_output=True, timeout=10)
                return result.returncode == 0
            
            elif self.system == "windows":
                # Use PowerShell to focus window
                ps_script = f"""
                Add-Type -AssemblyName System.Windows.Forms
                $window = [System.Windows.Forms.Application]::OpenForms | Where-Object {{ $_.Text -like "*{window_title}*" }}
                if ($window) {{
                    $window[0].Activate()
                    $true
                }} else {{
                    $false
                }}
                """
                
                result = subprocess.run(['powershell', '-Command', ps_script], 
                                      capture_output=True, text=True, timeout=10)
                return result.returncode == 0 and "True" in result.stdout
            
        except Exception as e:
            logging.error(f"Error focusing window {window_title}: {e}")
            return False
    
    def close_window(self, window_title: str) -> bool:
        """
        Close a window by title.
        
        Args:
            window_title: Title of the window to close
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.system == "linux":
                # Use wmctrl to close window
                result = subprocess.run(['wmctrl', '-c', window_title], 
                                      capture_output=True, timeout=10)
                return result.returncode == 0
            
            elif self.system == "windows":
                # Use PowerShell to close window
                ps_script = f"""
                Add-Type -AssemblyName System.Windows.Forms
                $window = [System.Windows.Forms.Application]::OpenForms | Where-Object {{ $_.Text -like "*{window_title}*" }}
                if ($window) {{
                    $window[0].Close()
                    $true
                }} else {{
                    $false
                }}
                """
                
                result = subprocess.run(['powershell', '-Command', ps_script], 
                                      capture_output=True, text=True, timeout=10)
                return result.returncode == 0 and "True" in result.stdout
            
        except Exception as e:
            logging.error(f"Error closing window {window_title}: {e}")
            return False
    
    def launch_application(self, app_name: str, args: Optional[List[str]] = None) -> bool:
        """
        Launch an application with OS-specific handling.
        
        Args:
            app_name: Application name or command
            args: Command line arguments
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # OS-specific application mappings
            os_apps = {
                "terminal": self.linux_config["terminal"] if self.system == "linux" else self.windows_config["terminal"],
                "file_manager": self.linux_config["file_manager"] if self.system == "linux" else self.windows_config["file_manager"],
                "text_editor": self.linux_config["text_editor"] if self.system == "linux" else self.windows_config["text_editor"],
                "calculator": self.linux_config["calculator"] if self.system == "linux" else self.windows_config["calculator"],
                "settings": self.linux_config["settings"] if self.system == "linux" else self.windows_config["settings"]
            }
            
            # Check if it's an OS-specific app
            if app_name.lower() in os_apps:
                app_name = os_apps[app_name.lower()]
            
            # Common application mappings
            app_mappings = {
                # Text editors
                "notepad": "notepad" if self.system == "windows" else "gedit",
                "textedit": "notepad" if self.system == "windows" else "gedit",
                "gedit": "gedit",
                "kate": "kate",
                "mousepad": "mousepad",
                "leafpad": "leafpad",
                
                # Office applications - these need special handling
                "writer": "libreoffice",
                "calc": "libreoffice", 
                "impress": "libreoffice",
                "draw": "libreoffice",
                "word": "libreoffice",
                "excel": "libreoffice",
                "powerpoint": "libreoffice",
                
                # Browsers
                "firefox": "firefox",
                "chrome": "google-chrome" if self.system == "linux" else "chrome",
                "chromium": "chromium",
                "edge": "microsoft-edge",
                
                # File managers
                "explorer": "explorer" if self.system == "windows" else "dolphin",
                "dolphin": "dolphin",
                "nautilus": "nautilus",
                "thunar": "thunar",
                "pcmanfm": "pcmanfm",
                
                # Terminals
                "terminal": "gnome-terminal" if self.system == "linux" else "cmd",
                "cmd": "cmd" if self.system == "windows" else "gnome-terminal",
                "powershell": "powershell" if self.system == "windows" else "gnome-terminal",
                
                # Media players
                "vlc": "vlc",
                "mpv": "mpv",
                "totem": "totem",
                "kodi": "kodi",
                
                # Development tools
                "code": "code",
                "vscode": "code",
                "sublime": "sublime-text",
                "atom": "atom",
                "gedit": "gedit",
                
                # System tools
                "calculator": "calc" if self.system == "windows" else "gnome-calculator",
                "calc": "calc" if self.system == "windows" else "gnome-calculator",
                "gnome-calculator": "gnome-calculator",
                "kcalc": "kcalc",
                
                # Settings
                "settings": "ms-settings:" if self.system == "windows" else "systemsettings",
                "control": "control" if self.system == "windows" else "systemsettings",
                "systemsettings": "systemsettings",
            }
            
            # Special handling for LibreOffice applications
            libreoffice_apps = {
                "writer": "--writer",
                "calc": "--calc",
                "impress": "--impress", 
                "draw": "--draw",
                "word": "--writer",
                "excel": "--calc",
                "powerpoint": "--impress"
            }
            
            # Check if it's a mapped application
            if app_name.lower() in app_mappings:
                mapped_app = app_mappings[app_name.lower()]
                if self.system == "windows" and mapped_app.startswith("ms-settings:"):
                    # Windows settings URL
                    import webbrowser
                    webbrowser.open(mapped_app)
                    logging.info(f"Opened Windows settings: {mapped_app}")
                    return True
                else:
                    app_name = mapped_app
                    
                    # Add LibreOffice arguments if needed
                    if app_name == "libreoffice" and app_name.lower() in libreoffice_apps:
                        if args is None:
                            args = []
                        args.insert(0, libreoffice_apps[app_name.lower()])
            
            # Try to find the application if it's not a full path
            if not os.path.isabs(app_name) and not app_name.startswith(('http://', 'https://')):
                # Check if it's in PATH
                if self.system == "linux":
                    result = subprocess.run(['which', app_name], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        app_name = result.stdout.strip()
                    else:
                        # Try common locations
                        common_paths = [
                            f"/usr/bin/{app_name}",
                            f"/usr/local/bin/{app_name}",
                            f"/opt/{app_name}/bin/{app_name}",
                            f"/snap/bin/{app_name}",
                            f"/usr/share/applications/{app_name}.desktop"
                        ]
                        for path in common_paths:
                            if os.path.exists(path):
                                app_name = path
                                break
                elif self.system == "windows":
                    # Windows: try common program files locations
                    program_files = os.environ.get('PROGRAMFILES', 'C:\\Program Files')
                    program_files_x86 = os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)')
                    
                    common_paths = [
                        os.path.join(program_files, app_name, f"{app_name}.exe"),
                        os.path.join(program_files_x86, app_name, f"{app_name}.exe"),
                        os.path.join(program_files, "Microsoft", app_name, f"{app_name}.exe"),
                        os.path.join(program_files_x86, "Microsoft", app_name, f"{app_name}.exe"),
                    ]
                    
                    for path in common_paths:
                        if os.path.exists(path):
                            app_name = path
                            break
            
            # Build command
            cmd = [app_name]
            if args:
                cmd.extend(args)
            
            logging.info(f"Launching application: {' '.join(cmd)}")
            
            # Launch the application
            if self.system == "windows":
                # Windows: use subprocess.Popen with shell=True for better compatibility
                subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                # Linux: use subprocess.Popen
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            logging.info(f"Successfully launched: {app_name}")
            return True
            
        except Exception as e:
            logging.error(f"Error launching application {app_name}: {e}")
            return False
    
    def get_screen_resolution(self) -> Optional[Tuple[int, int]]:
        """
        Get screen resolution.
        
        Returns:
            Tuple of (width, height) or None if error
        """
        try:
            if self.system == "linux":
                # Use xrandr to get screen info
                result = subprocess.run(['xrandr'], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    # Parse xrandr output for primary display
                    for line in result.stdout.split('\n'):
                        if '*' in line and 'primary' in line:
                            # Extract resolution from line like "1920x1080*"
                            import re
                            match = re.search(r'(\d+)x(\d+)', line)
                            if match:
                                return (int(match.group(1)), int(match.group(2)))
            
            elif self.system == "windows":
                # Use PowerShell to get screen resolution
                ps_script = """
                Add-Type -AssemblyName System.Windows.Forms
                $screen = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
                "$($screen.Width)x$($screen.Height)"
                """
                
                result = subprocess.run(['powershell', '-Command', ps_script], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    resolution = result.stdout.strip()
                    if 'x' in resolution:
                        width, height = resolution.split('x')
                        return (int(width), int(height))
            
        except Exception as e:
            logging.error(f"Error getting screen resolution: {e}")
        
        return None
    
    def list_screenshots(self) -> List[Dict[str, Any]]:
        """List all saved screenshots."""
        screenshots = []
        
        try:
            for screenshot_file in self.home_dir / "ratatoskr" / "screenshots".glob("*.png"):
                stat = screenshot_file.stat()
                screenshots.append({
                    "name": screenshot_file.name,
                    "path": str(screenshot_file),
                    "size": stat.st_size,
                    "created": stat.st_ctime,
                    "modified": stat.st_mtime
                })
            
            # Sort by creation time (newest first)
            screenshots.sort(key=lambda x: x["created"], reverse=True)
            
        except Exception as e:
            logging.error(f"Error listing screenshots: {e}")
        
        return screenshots
    
    def delete_screenshot(self, filename: str) -> bool:
        """
        Delete a screenshot file.
        
        Args:
            filename: Name of the screenshot file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            screenshot_path = self.home_dir / "ratatoskr" / "screenshots" / filename
            
            if not screenshot_path.exists():
                logging.error(f"Screenshot not found: {filename}")
                return False
            
            screenshot_path.unlink()
            logging.info(f"Deleted screenshot: {filename}")
            return True
            
        except Exception as e:
            logging.error(f"Error deleting screenshot {filename}: {e}")
            return False
    
    def list_menu_applications(self) -> dict:
        """
        List all applications available in the OS menu (Linux .desktop files).
        Returns a dict mapping app name to exec command.
        """
        app_dirs = [
            '/usr/share/applications/',
            '/usr/local/share/applications/',
            os.path.expanduser('~/.local/share/applications/')
        ]
        apps = {}
        for app_dir in app_dirs:
            for desktop_file in glob.glob(os.path.join(app_dir, '*.desktop')):
                try:
                    config = configparser.ConfigParser(interpolation=None)
                    config.read(desktop_file)
                    if 'Desktop Entry' in config:
                        entry = config['Desktop Entry']
                        name = entry.get('Name')
                        exec_cmd = entry.get('Exec')
                        if name and exec_cmd and not entry.get('NoDisplay', 'false').lower() == 'true':
                            # Remove field codes (like %U, %f, etc.)
                            exec_cmd = shlex.split(exec_cmd)[0]
                            apps[name] = exec_cmd
                except Exception as e:
                    logging.debug(f"Error parsing {desktop_file}: {e}")
        return apps

    def launch_menu_application(self, app_query: str) -> bool:
        """
        Launch an application from the OS menu by name (fuzzy match).
        Args:
            app_query: Application name (user input)
        Returns:
            True if successful, False otherwise
        """
        apps = self.list_menu_applications()
        if not apps:
            logging.error("No menu applications found.")
            return False
        # Fuzzy match
        matches = get_close_matches(app_query, apps.keys(), n=1, cutoff=0.6)
        if not matches:
            logging.error(f"No application found matching '{app_query}'")
            return False
        app_name = matches[0]
        exec_cmd = apps[app_name]
        try:
            logging.info(f"Launching menu application: {app_name} -> {exec_cmd}")
            subprocess.Popen([exec_cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception as e:
            logging.error(f"Error launching menu application {app_name}: {e}")
            return False

# Global screen controller instance
screen_controller = ScreenController() 