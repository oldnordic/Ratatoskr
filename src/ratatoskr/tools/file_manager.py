"""
Cross-platform file system manager for Ratatoskr.

This module provides file system operations that work on both Windows and Linux,
including file search, navigation, and basic file operations.
"""

import os
import platform
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import fnmatch

class FileManager:
    """
    Cross-platform file system manager.
    
    Provides file operations, search, and navigation capabilities
    that work consistently across Windows and Linux.
    """
    
    def __init__(self):
        self.system = platform.system().lower()
        self.home_dir = Path.home()
        self.current_dir = self.home_dir
        self.search_index = {}
        
        logging.info(f"FileManager initialized for {self.system}")
        logging.info(f"Home directory: {self.home_dir}")
    
    def get_system_info(self) -> Dict[str, str]:
        """Get system information for file operations."""
        return {
            "system": self.system,
            "home_dir": str(self.home_dir),
            "current_dir": str(self.current_dir),
            "separator": os.sep
        }
    
    def list_directory(self, path: Optional[str] = None, show_hidden: bool = False) -> List[Dict]:
        """
        List contents of a directory.
        
        Args:
            path: Directory path (None for current directory)
            show_hidden: Whether to show hidden files
            
        Returns:
            List of file/directory information dictionaries
        """
        try:
            target_path = Path(path) if path else self.current_dir
            target_path = target_path.resolve()
            
            if not target_path.exists():
                raise FileNotFoundError(f"Directory not found: {target_path}")
            
            if not target_path.is_dir():
                raise NotADirectoryError(f"Path is not a directory: {target_path}")
            
            items = []
            
            for item in target_path.iterdir():
                # Skip hidden files unless requested
                if not show_hidden and item.name.startswith('.'):
                    continue
                
                try:
                    stat = item.stat()
                    item_info = {
                        "name": item.name,
                        "path": str(item),
                        "is_dir": item.is_dir(),
                        "is_file": item.is_file(),
                        "size": stat.st_size if item.is_file() else 0,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        "permissions": oct(stat.st_mode)[-3:]
                    }
                    items.append(item_info)
                except (OSError, PermissionError) as e:
                    logging.warning(f"Error accessing {item}: {e}")
                    continue
            
            # Sort: directories first, then files, both alphabetically
            items.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
            
            return items
            
        except Exception as e:
            logging.error(f"Error listing directory {path}: {e}")
            return []
    
    def navigate_to(self, path: str) -> bool:
        """
        Navigate to a directory.
        
        Args:
            path: Directory path to navigate to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            target_path = Path(path).resolve()
            
            if not target_path.exists():
                logging.error(f"Path does not exist: {target_path}")
                return False
            
            if not target_path.is_dir():
                logging.error(f"Path is not a directory: {target_path}")
                return False
            
            self.current_dir = target_path
            logging.info(f"Navigated to: {self.current_dir}")
            return True
            
        except Exception as e:
            logging.error(f"Error navigating to {path}: {e}")
            return False
    
    def search_files(self, query: str, search_path: Optional[str] = None, 
                    file_types: Optional[List[str]] = None, 
                    case_sensitive: bool = False) -> List[Dict]:
        """
        Search for files by name or content.
        
        Args:
            query: Search query
            search_path: Path to search in (None for current directory)
            file_types: List of file extensions to search (e.g., ['.txt', '.py'])
            case_sensitive: Whether search is case sensitive
            
        Returns:
            List of matching files
        """
        try:
            search_path = Path(search_path) if search_path else self.current_dir
            search_path = search_path.resolve()
            
            if not search_path.exists():
                return []
            
            matches = []
            query_lower = query if case_sensitive else query.lower()
            
            for root, dirs, files in os.walk(search_path):
                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                for file in files:
                    # Skip hidden files
                    if file.startswith('.'):
                        continue
                    
                    # Check file type filter
                    if file_types:
                        file_ext = Path(file).suffix.lower()
                        if file_ext not in file_types:
                            continue
                    
                    file_path = Path(root) / file
                    
                    # Check filename match
                    filename_match = False
                    if case_sensitive:
                        filename_match = query in file
                    else:
                        filename_match = query_lower in file.lower()
                    
                    if filename_match:
                        try:
                            stat = file_path.stat()
                            match_info = {
                                "name": file,
                                "path": str(file_path),
                                "size": stat.st_size,
                                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                "match_type": "filename"
                            }
                            matches.append(match_info)
                        except (OSError, PermissionError):
                            continue
            
            return matches
            
        except Exception as e:
            logging.error(f"Error searching files: {e}")
            return []
    
    def read_file(self, file_path: str, encoding: str = 'utf-8') -> Optional[str]:
        """
        Read a text file.
        
        Args:
            file_path: Path to the file
            encoding: File encoding
            
        Returns:
            File content or None if error
        """
        try:
            path = Path(file_path).resolve()
            
            if not path.exists():
                logging.error(f"File not found: {path}")
                return None
            
            if not path.is_file():
                logging.error(f"Path is not a file: {path}")
                return None
            
            with open(path, 'r', encoding=encoding) as f:
                content = f.read()
            
            logging.info(f"Successfully read file: {path}")
            return content
            
        except UnicodeDecodeError:
            logging.error(f"Encoding error reading file: {file_path}")
            return None
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
            return None
    
    def write_file(self, file_path: str, content: str, encoding: str = 'utf-8') -> bool:
        """
        Write content to a file.
        
        Args:
            file_path: Path to the file
            content: Content to write
            encoding: File encoding
            
        Returns:
            True if successful, False otherwise
        """
        try:
            path = Path(file_path).resolve()
            
            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding=encoding) as f:
                f.write(content)
            
            logging.info(f"Successfully wrote file: {path}")
            return True
            
        except Exception as e:
            logging.error(f"Error writing file {file_path}: {e}")
            return False
    
    def copy_file(self, source: str, destination: str) -> bool:
        """
        Copy a file.
        
        Args:
            source: Source file path
            destination: Destination file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            source_path = Path(source).resolve()
            dest_path = Path(destination).resolve()
            
            if not source_path.exists():
                logging.error(f"Source file not found: {source_path}")
                return False
            
            # Create destination directory if it doesn't exist
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(source_path, dest_path)
            logging.info(f"Successfully copied {source_path} to {dest_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error copying file {source} to {destination}: {e}")
            return False
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file.
        
        Args:
            file_path: Path to the file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            path = Path(file_path).resolve()
            
            if not path.exists():
                logging.error(f"File not found: {path}")
                return False
            
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            
            logging.info(f"Successfully deleted: {path}")
            return True
            
        except Exception as e:
            logging.error(f"Error deleting file {file_path}: {e}")
            return False
    
    def get_file_info(self, file_path: str) -> Optional[Dict]:
        """
        Get detailed information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File information dictionary or None if error
        """
        try:
            path = Path(file_path).resolve()
            
            if not path.exists():
                return None
            
            stat = path.stat()
            
            info = {
                "name": path.name,
                "path": str(path),
                "is_dir": path.is_dir(),
                "is_file": path.is_file(),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
                "permissions": oct(stat.st_mode)[-3:],
                "extension": path.suffix.lower() if path.is_file() else "",
                "parent": str(path.parent)
            }
            
            return info
            
        except Exception as e:
            logging.error(f"Error getting file info for {file_path}: {e}")
            return None
    
    def create_directory(self, dir_path: str) -> bool:
        """
        Create a directory.
        
        Args:
            dir_path: Path to the directory to create
            
        Returns:
            True if successful, False otherwise
        """
        try:
            path = Path(dir_path).resolve()
            
            if path.exists():
                logging.warning(f"Directory already exists: {path}")
                return True
            
            path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Successfully created directory: {path}")
            return True
            
        except Exception as e:
            logging.error(f"Error creating directory {dir_path}: {e}")
            return False

# Global file manager instance
file_manager = FileManager() 