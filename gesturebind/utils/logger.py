"""
Logging utility for GestureBind

Sets up and configures application logging.
"""

import os
import logging
import sys
from datetime import datetime
from pathlib import Path

def setup_logger():
    """
    Configure the application logger.
    
    Returns:
        logging.Logger: Configured root logger
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add formatter to console handler
    console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    # Create log directory if it doesn't exist
    log_dir = Path.home() / ".gesturebind" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create file handler
    log_filename = f"gesturebind_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = log_dir / log_filename
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)  # Debug level for file logs
    file_handler.setFormatter(formatter)
    
    # Add file handler to logger
    logger.addHandler(file_handler)
    
    # Log initial message
    logger.info(f"Logger initialized. Log file: {log_path}")
    
    return logger

def set_log_level(level):
    """
    Set the log level for the application logger.
    
    Args:
        level (str): Log level name (debug, info, warning, error, critical)
        
    Returns:
        bool: Success or failure
    """
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }
    
    if level.lower() not in level_map:
        return False
        
    logging.getLogger().setLevel(level_map[level.lower()])
    logging.info(f"Log level set to: {level}")
    return True

def get_recent_logs(n=100):
    """
    Get the most recent log entries.
    
    Args:
        n (int): Maximum number of log entries to retrieve
        
    Returns:
        list: Recent log entries
    """
    log_dir = Path.home() / ".gesturebind" / "logs"
    
    if not log_dir.exists():
        return []
    
    # Find most recent log file
    log_files = sorted(log_dir.glob("gesturebind_*.log"), key=os.path.getmtime, reverse=True)
    
    if not log_files:
        return []
    
    recent_log_file = log_files[0]
    
    # Read the last n lines
    entries = []
    try:
        with open(recent_log_file, 'r') as f:
            lines = f.readlines()
            entries = lines[-n:] if len(lines) > n else lines
    except Exception as e:
        logging.error(f"Error reading log file: {e}")
    
    return entries