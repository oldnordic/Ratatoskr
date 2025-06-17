"""
LibreOffice integration controller for Ratatoskr.

This module provides LibreOffice document creation, editing, and management
capabilities through LibreOffice's command-line interface and Python-UNO bridge.
"""

import os
import platform
import subprocess
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Any
import time

class LibreOfficeController:
    """
    Controller for LibreOffice operations.
    
    Provides document creation, editing, and management capabilities
    through LibreOffice's command-line interface.
    """
    
    def __init__(self):
        self.system = platform.system().lower()
        self.libreoffice_path = self._find_libreoffice()
        self.temp_dir = tempfile.mkdtemp(prefix="ratatoskr_libreoffice_")
        
        logging.info(f"LibreOfficeController initialized for {self.system}")
        if self.libreoffice_path:
            logging.info(f"LibreOffice found at: {self.libreoffice_path}")
        else:
            logging.warning("LibreOffice not found")
    
    def _find_libreoffice(self) -> Optional[str]:
        """Find LibreOffice installation path."""
        possible_paths = []
        
        if self.system == "linux":
            # Common Linux paths
            possible_paths = [
                "/usr/bin/libreoffice",
                "/usr/bin/soffice",
                "/opt/libreoffice/program/soffice",
                "/usr/local/bin/libreoffice",
                "/usr/local/bin/soffice"
            ]
        elif self.system == "windows":
            # Common Windows paths
            program_files = os.environ.get('PROGRAMFILES', 'C:\\Program Files')
            program_files_x86 = os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)')
            
            possible_paths = [
                os.path.join(program_files, "LibreOffice", "program", "soffice.exe"),
                os.path.join(program_files_x86, "LibreOffice", "program", "soffice.exe"),
                os.path.join(program_files, "LibreOffice*", "program", "soffice.exe"),
                os.path.join(program_files_x86, "LibreOffice*", "program", "soffice.exe")
            ]
        
        # Check each possible path
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Try to find via command line
        try:
            if self.system == "linux":
                result = subprocess.run(['which', 'libreoffice'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return result.stdout.strip()
                
                result = subprocess.run(['which', 'soffice'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return result.stdout.strip()
            elif self.system == "windows":
                result = subprocess.run(['where', 'soffice'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return result.stdout.strip().split('\n')[0]
        except Exception as e:
            logging.debug(f"Error finding LibreOffice via command line: {e}")
        
        return None
    
    def is_available(self) -> bool:
        """Check if LibreOffice is available."""
        return self.libreoffice_path is not None
    
    def get_info(self) -> Dict[str, Any]:
        """Get LibreOffice information."""
        info = {
            "available": self.is_available(),
            "system": self.system,
            "path": self.libreoffice_path,
            "temp_dir": self.temp_dir
        }
        
        if self.is_available():
            try:
                # Get version information
                result = subprocess.run([self.libreoffice_path, '--version'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    info["version"] = result.stdout.strip()
                else:
                    info["version"] = "Unknown"
            except Exception as e:
                logging.warning(f"Error getting LibreOffice version: {e}")
                info["version"] = "Error"
        
        return info
    
    def create_document(self, file_path: str, document_type: str = "writer") -> bool:
        """
        Create a new LibreOffice document.
        
        Args:
            file_path: Path where to save the document
            document_type: Type of document ("writer", "calc", "impress", "draw")
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logging.error("LibreOffice not available")
            return False
        
        try:
            # Map document types to file extensions
            extensions = {
                "writer": ".odt",
                "calc": ".ods", 
                "impress": ".odp",
                "draw": ".odg"
            }
            
            if document_type not in extensions:
                logging.error(f"Unknown document type: {document_type}")
                return False
            
            # Ensure file has correct extension
            file_path = Path(file_path)
            if not file_path.suffix:
                file_path = file_path.with_suffix(extensions[document_type])
            
            # Create empty document
            cmd = [
                self.libreoffice_path,
                '--headless',
                '--convert-to', extensions[document_type][1:],  # Remove dot
                '--outdir', str(file_path.parent),
                '--infilter="Text (encoded):UTF8"',
                '--accept=socket,host=localhost,port=2002;urp;StarOffice.ServiceManager'
            ]
            
            # Create a temporary empty file to convert
            temp_file = Path(self.temp_dir) / f"temp_{document_type}.txt"
            temp_file.write_text("", encoding='utf-8')
            
            cmd.append(str(temp_file))
            
            logging.info(f"Creating {document_type} document: {file_path}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logging.info(f"Successfully created document: {file_path}")
                return True
            else:
                logging.error(f"Error creating document: {result.stderr}")
                return False
                
        except Exception as e:
            logging.error(f"Error creating LibreOffice document: {e}")
            return False
    
    def open_document(self, file_path: str) -> bool:
        """
        Open a document in LibreOffice.
        
        Args:
            file_path: Path to the document to open
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logging.error("LibreOffice not available")
            return False
        
        try:
            file_path = Path(file_path).resolve()
            
            if not file_path.exists():
                logging.error(f"Document not found: {file_path}")
                return False
            
            cmd = [self.libreoffice_path, str(file_path)]
            
            logging.info(f"Opening document: {file_path}")
            # Run in background
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            return True
            
        except Exception as e:
            logging.error(f"Error opening LibreOffice document: {e}")
            return False
    
    def convert_document(self, input_path: str, output_path: str, 
                        output_format: str = "pdf") -> bool:
        """
        Convert a document to another format.
        
        Args:
            input_path: Path to input document
            output_path: Path for output document
            output_format: Output format (pdf, docx, xlsx, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logging.error("LibreOffice not available")
            return False
        
        try:
            input_path = Path(input_path).resolve()
            output_path = Path(output_path).resolve()
            
            if not input_path.exists():
                logging.error(f"Input document not found: {input_path}")
                return False
            
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            cmd = [
                self.libreoffice_path,
                '--headless',
                '--convert-to', output_format,
                '--outdir', str(output_path.parent),
                str(input_path)
            ]
            
            logging.info(f"Converting {input_path} to {output_format}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                logging.info(f"Successfully converted document to {output_path}")
                return True
            else:
                logging.error(f"Error converting document: {result.stderr}")
                return False
                
        except Exception as e:
            logging.error(f"Error converting LibreOffice document: {e}")
            return False
    
    def extract_text(self, file_path: str) -> Optional[str]:
        """
        Extract text from a LibreOffice document.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Extracted text or None if error
        """
        if not self.is_available():
            logging.error("LibreOffice not available")
            return None
        
        try:
            file_path = Path(file_path).resolve()
            
            if not file_path.exists():
                logging.error(f"Document not found: {file_path}")
                return None
            
            # Convert to text format first
            temp_dir = Path(self.temp_dir)
            temp_text_file = temp_dir / f"extracted_{file_path.stem}.txt"
            
            cmd = [
                self.libreoffice_path,
                '--headless',
                '--convert-to', 'txt',
                '--outdir', str(temp_dir),
                str(file_path)
            ]
            
            logging.info(f"Extracting text from: {file_path}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and temp_text_file.exists():
                text = temp_text_file.read_text(encoding='utf-8')
                temp_text_file.unlink()  # Clean up
                return text
            else:
                logging.error(f"Error extracting text: {result.stderr}")
                return None
                
        except Exception as e:
            logging.error(f"Error extracting text from LibreOffice document: {e}")
            return None
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get supported input and output formats."""
        return {
            "input": [
                ".odt", ".ods", ".odp", ".odg",  # OpenDocument formats
                ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",  # Microsoft formats
                ".rtf", ".txt", ".csv", ".html", ".xml"  # Other formats
            ],
            "output": [
                "pdf", "docx", "xlsx", "pptx",  # Common formats
                "odt", "ods", "odp", "odg",  # OpenDocument formats
                "rtf", "txt", "csv", "html"  # Text formats
            ]
        }
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file (wrapper for file manager).
        
        Args:
            file_path: Path to the file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from .file_manager import file_manager
            return file_manager.delete_file(file_path)
        except Exception as e:
            logging.error(f"Error deleting file {file_path}: {e}")
            return False
    
    def cleanup(self):
        """Clean up temporary files."""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logging.info("Cleaned up LibreOffice temporary directory")
        except Exception as e:
            logging.warning(f"Error cleaning up LibreOffice temp directory: {e}")

# Global LibreOffice controller instance
libreoffice_controller = LibreOfficeController() 