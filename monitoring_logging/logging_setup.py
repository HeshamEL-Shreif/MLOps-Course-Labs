from logging import StreamHandler, Formatter, getLogger, DEBUG
from logging import INFO, WARNING, ERROR, CRITICAL
from colorama import Fore, Style


class CustomFormatter(Formatter):
    """Custom formatter to add colors to log messages based on their level."""

    COLORS = {
        DEBUG: Fore.CYAN,
        INFO: Fore.GREEN,
        WARNING: Fore.YELLOW,
        ERROR: Fore.RED,
        CRITICAL: Fore.MAGENTA,
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelno, Style.RESET_ALL)
        message = super().format(record)
        return f"{log_color}{message}{Style.RESET_ALL}"
    
def setup_logging():
    """Set up logging with a custom formatter and stream handler."""
    logger = getLogger()
    logger.setLevel(DEBUG)
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    handler = StreamHandler()
    handler.setLevel(DEBUG)

    formatter = CustomFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger

