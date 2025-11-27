import logging
import sys
from typing import Optional

class PipelineLogger:
    """
    Centralized logger for pipeline actors that supports both file logging and console output.
    Provides colored output for console and timestamped, method-tagged logs.
    """
    
    # ANSI color codes for console output
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green  
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def __init__(self, 
                 name: str = "pipeline",
                 log_file: Optional[str] = None,
                 console_level: str = "INFO",
                 file_level: str = "INFO",
                 use_colors: bool = True):
        """
        Initialize pipeline logger.
        
        Args:
            name: Logger name
            log_file: Path to log file (if None, only console logging)
            console_level: Minimum level for console output
            file_level: Minimum level for file output
            use_colors: Whether to use colored console output
        """
        self.name = name
        self.log_file = log_file
        self.console_level = console_level
        self.file_level = file_level
        self.use_colors = use_colors
        
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup the actual logging infrastructure."""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        self.use_colors_enabled = self.use_colors and sys.stdout.isatty()
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.console_level.upper()))
        
        if self.use_colors_enabled:
            console_formatter = ColoredFormatter(
                fmt='%(asctime)s | %(method)8s | %(levelname)4s | %(message)s',
                datefmt='%H:%M:%S'
            )
        else:
            console_formatter = logging.Formatter(
                fmt='%(asctime)s | %(method)8s | %(levelname)4s | %(message)s',
                datefmt='%H:%M:%S'
            )
        
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if specified)
        if self.log_file:
            self._add_file_handler()
    
    def _add_file_handler(self):
        """Add file handler with current log_file path."""
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(getattr(logging, self.file_level.upper()))
        file_formatter = logging.Formatter(
            fmt='%(asctime)s | %(method)8s | %(levelname)4s | %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    def _remove_file_handlers(self):
        """Remove all existing file handlers."""
        handlers_to_remove = [h for h in self.logger.handlers if isinstance(h, logging.FileHandler)]
        for handler in handlers_to_remove:
            handler.close()  # Important: close the handler to release file resources
            self.logger.removeHandler(handler)
    
    def update_logger(self, name: str = None, log_file: str = None):
        """Update logger configuration and recreate handlers."""
        config_changed = False
        
        # Update name if provided
        if name and name != self.name:
            self.name = name
            # Create a new logger instance with the new name
            old_handlers = self.logger.handlers.copy()
            self.logger = logging.getLogger(self.name)
            self.logger.setLevel(logging.DEBUG)
            
            # Copy handlers to new logger (except file handlers, we'll recreate those)
            for handler in old_handlers:
                if not isinstance(handler, logging.FileHandler):
                    self.logger.addHandler(handler)
            
            config_changed = True
        
        # Update log file if provided
        if log_file and log_file != self.log_file:
            self.log_file = log_file
            config_changed = True
        
        # If file logging is needed, remove old file handlers and add new one
        if self.log_file:
            self._remove_file_handlers()
            self._add_file_handler()
        
        return config_changed
        
    def log(self, level: str, method: str, message: str, *args, **kwargs):
        """Internal logging method that adds method context."""
        # Create a LogRecord with custom 'method' field
        extra = {'method': method}
        getattr(self.logger, level.lower())(message, *args, extra=extra, **kwargs)
    
class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to console output."""

    # 4-letter abbreviations for consistent spacing
    LEVEL_ABBREVIATIONS = {
        'DEBUG': 'DBUG',
        'INFO': 'INFO',
        'WARNING': 'WARN',
        'ERROR': 'ERRO',
        'CRITICAL': 'CRIT'
    }

    def format(self, record):
        # Abbreviate log level to 4 characters for consistent spacing
        original_levelname = record.levelname
        record.levelname = self.LEVEL_ABBREVIATIONS.get(original_levelname, original_levelname[:4])

        # Get the color for this level
        color = PipelineLogger.COLORS.get(original_levelname, '')
        reset = PipelineLogger.COLORS['RESET']

        # Format the message
        formatted = super().format(record)

        # Add color to the entire line
        if color:
            formatted = f"{color}{formatted}{reset}"

        # Restore original level name
        record.levelname = original_levelname

        return formatted