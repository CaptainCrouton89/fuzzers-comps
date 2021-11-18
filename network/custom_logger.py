import os
import sys
import logging
from datetime import datetime

# Configure logger
def init_logger(log_path, level, config_path):
    os.makedirs(log_path, exist_ok=True)

    width = os.get_terminal_size().columns

    formatter = logging.Formatter('%(levelname)s: %(message)s \t\t@ %(filename)s: %(funcName)s: %(lineno)d', datefmt='%m/%d/%Y %I:%M:%S',)
    logging.getLogger().setLevel(logging.DEBUG)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(getattr(logging, level.upper(), None))

    file_handler = logging.FileHandler(filename=f'{log_path}/{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.log', mode='w')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    logging.basicConfig(
        handlers=[file_handler, stdout_handler])
        
    logging.info(f"Running pipeline with {config_path}")