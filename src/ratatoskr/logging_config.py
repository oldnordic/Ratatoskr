"""
Advanced logging configuration for the Ratatoskr AI assistant.

This module provides comprehensive logging setup with configurable levels,
formatters, handlers, and rotation capabilities. It integrates with the
main configuration system for centralized settings management.

Features:
- Configurable log levels and formats
- File and console logging
- Log rotation and size limits
- Structured logging support
- Performance monitoring
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from config import config_manager


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    max_log_size: Optional[int] = None
) -> None:
    """
    Configure comprehensive logging for the application.
    Sets up logging with both file and console handlers, configurable levels, and rotation capabilities.
    Uses config_manager for default values but allows runtime overrides.
    """
    level = log_level or config_manager.get('logging.log_level', 'INFO')
    log_file_path = log_file or config_manager.get('logging.log_file', 'application.log')
    format_string = log_format or config_manager.get('logging.log_format', '%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    max_size = max_log_size or config_manager.get('logging.max_log_size', 10485760)

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    logger.handlers.clear()
    formatter = logging.Formatter(format_string)
    file_handler = _create_file_handler(log_file_path, max_size, formatter)
    file_handler.setLevel(numeric_level)
    logger.addHandler(file_handler)
    console_handler = _create_console_handler(formatter)
    console_handler.setLevel(numeric_level)
    logger.addHandler(console_handler)
    logging.info(f"Logging configured successfully - Level: {level}, File: {log_file_path}")


def _create_file_handler(log_file: str, max_size: int, formatter: logging.Formatter) -> logging.Handler:
    """
    Create a file handler with rotation capabilities.
    
    Args:
        log_file: Path to the log file
        max_size: Maximum file size in bytes
        formatter: Log formatter to use
        
    Returns:
        logging.Handler: Configured file handler
    """
    try:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_size,
            backupCount=5,  # Keep 5 backup files
            encoding='utf-8'
        )
        handler.setFormatter(formatter)
        return handler
        
    except Exception as e:
        # Fallback to basic file handler if rotation fails
        logging.warning(f"Failed to create rotating file handler: {e}, using basic handler")
        handler = logging.FileHandler(log_file, mode="w", encoding='utf-8')
        handler.setFormatter(formatter)
        return handler


def _create_console_handler(formatter: logging.Formatter) -> logging.Handler:
    """
    Create a console handler for terminal output.
    
    Args:
        formatter: Log formatter to use
        
    Returns:
        logging.Handler: Configured console handler
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    return handler


def setup_structured_logging() -> None:
    """
    Set up structured logging with JSON format for better parsing.
    
    This function configures logging to output structured JSON logs
    that are easier to parse and analyze programmatically.
    """
    import json
    from datetime import datetime
    
    class JSONFormatter(logging.Formatter):
        """Custom JSON formatter for structured logging."""
        
        def format(self, record: logging.LogRecord) -> str:
            """Format log record as JSON."""
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
                "message": record.getMessage()
            }
            
            # Add exception info if present
            if record.exc_info:
                log_entry["exception"] = self.formatException(record.exc_info)
            
            # Add extra fields if present
            if hasattr(record, 'extra_fields'):
                log_entry.update(record.extra_fields)
            
            return json.dumps(log_entry)
    
    # Set up JSON logging
    setup_logging(
        log_format="%(message)s",  # JSON formatter handles the rest
        log_file="application.json.log"
    )
    
    # Replace formatter with JSON formatter
    logger = logging.getLogger()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setFormatter(JSONFormatter())


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (usually module name)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)


def log_performance(func):
    """
    Decorator to log function performance metrics.
    
    Args:
        func: Function to wrap with performance logging
        
    Returns:
        Wrapped function with performance logging
    """
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger = logging.getLogger(func.__module__)
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {execution_time:.4f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.4f}s: {e}")
            raise
    
    return wrapper


def log_with_context(context: dict):
    """
    Decorator to add context information to log messages.
    
    Args:
        context: Dictionary of context information to add to logs
        
    Returns:
        Decorator function
    """
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            
            # Add context to log record
            old_factory = logging.getLogRecordFactory()
            
            def record_factory(*args, **kwargs):
                record = old_factory(*args, **kwargs)
                record.extra_fields = context
                return record
            
            logging.setLogRecordFactory(record_factory)
            
            try:
                return func(*args, **kwargs)
            finally:
                # Restore original factory
                logging.setLogRecordFactory(old_factory)
        
        return wrapper
    return decorator


# Initialize logging when module is imported
if not logging.getLogger().handlers:
    setup_logging()
