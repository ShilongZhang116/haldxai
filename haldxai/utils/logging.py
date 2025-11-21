# -*- coding: utf-8 -*-
"""
Logging utilities for HALDxAI platform.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler


def setup_logging(
    log_level: str = "INFO",
    log_format: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> None:
    """Setup logging configuration for HALDxAI platform.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Custom log format string
        log_file: Path to log file (optional)
        enable_console: Enable console logging
        enable_file: Enable file logging
        max_file_size: Maximum log file size in bytes
        backup_count: Number of backup files to keep
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Default log format
    if log_format is None:
        log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if enable_file:
        if log_file is None:
            # Default log file location
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / "haldxai.log"
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels to reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__module__ + "." + self.__class__.__name__)


def log_function_call(func):
    """Decorator to log function calls with arguments and return value."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned: {type(result)}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} raised: {type(e).__name__}: {e}")
            raise
    
    return wrapper


def log_method_call(method):
    """Decorator to log method calls with self instance."""
    def wrapper(self, *args, **kwargs):
        logger = get_logger(self.__class__.__module__)
        class_name = self.__class__.__name__
        logger.debug(f"Calling {class_name}.{method.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = method(self, *args, **kwargs)
            logger.debug(f"{class_name}.{method.__name__} returned: {type(result)}")
            return result
        except Exception as e:
            logger.error(f"{class_name}.{method.__name__} raised: {type(e).__name__}: {e}")
            raise
    
    return wrapper


class ProgressLogger:
    """Logger for tracking progress of long-running operations."""
    
    def __init__(self, logger: logging.Logger, total: int, description: str = "Processing"):
        self.logger = logger
        self.total = total
        self.current = 0
        self.description = description
        self.last_percent = -1
    
    def update(self, increment: int = 1, message: str = "") -> None:
        """Update progress counter."""
        self.current += increment
        percent = (self.current / self.total) * 100
        
        # Log every 10% or at completion
        if percent - self.last_percent >= 10 or self.current >= self.total:
            self.logger.info(f"{self.description}: {self.current}/{self.total} ({percent:.1f}%) {message}")
            self.last_percent = percent
    
    def finish(self, message: str = "Completed") -> None:
        """Mark operation as finished."""
        self.logger.info(f"{self.description}: {message} (Total: {self.total})")


class StructuredLogger:
    """Logger for structured (JSON) logging."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_event(self, event: str, level: str = "INFO", **kwargs) -> None:
        """Log a structured event."""
        import json
        log_data = {
            "event": event,
            "level": level,
            **kwargs
        }
        
        log_message = json.dumps(log_data, default=str)
        getattr(self.logger, level.lower())(log_message)
    
    def log_error(self, error: Exception, context: str = "", **kwargs) -> None:
        """Log an error with context."""
        self.log_event(
            "error",
            level="ERROR",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context,
            **kwargs
        )
    
    def log_performance(self, operation: str, duration: float, **kwargs) -> None:
        """Log performance metrics."""
        self.log_event(
            "performance",
            operation=operation,
            duration=duration,
            **kwargs
        )


# Context manager for temporary logging level changes
class LogLevelContext:
    """Context manager for temporarily changing log level."""
    
    def __init__(self, logger: logging.Logger, level: str):
        self.logger = logger
        self.new_level = getattr(logging, level.upper())
        self.old_level = None
    
    def __enter__(self):
        self.old_level = self.logger.level
        self.logger.setLevel(self.new_level)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)


# Utility functions for common logging patterns
def log_system_info(logger: logging.Logger) -> None:
    """Log system information."""
    import platform
    import sys
    
    logger.info("System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python: {sys.version}")
    logger.info(f"  Working directory: {Path.cwd()}")


def log_package_info(logger: logging.Logger, package_name: str) -> None:
    """Log package version information."""
    try:
        import importlib.metadata as metadata
        version = metadata.version(package_name)
        logger.info(f"{package_name}: {version}")
    except Exception:
        logger.warning(f"Could not determine version of {package_name}")


def log_config_summary(logger: logging.Logger, config_dict: dict) -> None:
    """Log configuration summary (without sensitive data)."""
    logger.info("Configuration Summary:")
    
    def safe_log_config(data, prefix=""):
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                safe_log_config(value, full_key)
            elif any(sensitive in key.lower() for sensitive in ["key", "password", "secret", "token"]):
                logger.info(f"  {full_key}: ***")
            else:
                logger.info(f"  {full_key}: {value}")
    
    safe_log_config(config_dict)