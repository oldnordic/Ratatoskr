"""Utility for configuring application logging."""

import logging
import sys


def setup_logging() -> None:
    """Configure logging to write to ``application.log`` and stdout."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(module)s - %(message)s'
    )

    # Log everything to ``application.log``.  The file is overwritten on each
    # run so logs don't grow unbounded.
    file_handler = logging.FileHandler("application.log", mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Also emit logs to the console for immediate feedback.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Indicate that logging is ready for use.
    logging.info("Logging configured.")
