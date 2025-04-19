import logging
import os

def setup_logger(cfg):
    logger = logging.getLogger(cfg.logger.name)
    logger.setLevel(getattr(logging, cfg.logger.log_level.upper(), logging.INFO))

    # Avoid duplicate handlers
    if logger.hasHandlers():
        return logger

    # Create logs directory if not exists
    os.makedirs(os.path.dirname(cfg.logger.log_file), exist_ok=True)

    # File handler
    file_handler = logging.FileHandler(cfg.logger.log_file)
    file_handler.setLevel(getattr(logging, cfg.logger.log_level.upper(), logging.INFO))

    # Formatter
    formatter = logging.Formatter(cfg.logger.log_format)
    file_handler.setFormatter(formatter)

    # Console handler (optional)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Attach handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
