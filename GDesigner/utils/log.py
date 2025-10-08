#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from loguru import logger
from GDesigner.utils.const import GDesigner_ROOT

def configure_logging(print_level: str = "INFO", logfile_level: str = "DEBUG") -> None:
    logger.remove()
    logger.add(sys.stderr, level=print_level)
    logger.add(GDesigner_ROOT / 'logs/log.txt', level=logfile_level, rotation="10 MB")

def initialize_log_file(experiment_name: str, time_stamp: str) -> Path:
    try:
        log_file_path = GDesigner_ROOT / f'result/{experiment_name}/logs/log_{time_stamp}.txt'
        os.makedirs(log_file_path.parent, exist_ok=True)
        with open(log_file_path, 'w') as file:
            file.write("============ Start ============\n")
    except OSError as error:
        logger.error(f"Error initializing log file: {error}")
        raise
    return log_file_path

def swarmlog(sender: str, text: str, cost: float,  prompt_tokens: int, complete_tokens: int, log_file_path: str) -> None:
    formatted_message = (
        f"{sender} | 💵Total Cost: ${cost:.5f} | "
        f"Prompt Tokens: {prompt_tokens} | "
        f"Completion Tokens: {complete_tokens} | \n {text}"
    )
    logger.info(formatted_message)

    try:
        os.makedirs(log_file_path.parent, exist_ok=True)
        with open(log_file_path, 'a') as file:
            file.write(f"{formatted_message}\n")
    except OSError as error:
        logger.error(f"Error initializing log file: {error}")
        raise


def main():
    configure_logging()
    swarmlog("SenderName", "This is a test message.", 0.123)

if __name__ == "__main__":
    main()

