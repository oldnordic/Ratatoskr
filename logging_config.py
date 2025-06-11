import logging
import sys

def setup_logging():
    """Sets up logging to both a file and the console."""
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the lowest level to capture

    # Create a formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(module)s - %(message)s'
    )

    # --- File Handler ---
    # This handler writes logs to 'application.log'
    file_handler = logging.FileHandler('application.log', mode='w') # 'w' for overwrite each run
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # --- Console Handler ---
    # This handler prints logs to the console/terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logging.info("Logging configured.")