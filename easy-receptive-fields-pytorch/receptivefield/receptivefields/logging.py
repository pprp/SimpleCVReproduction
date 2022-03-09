import logging.config
from logging import Logger
from os import path

log_file_path = path.join(path.dirname(path.abspath(__file__)), "resources/logger.conf")
logging.config.fileConfig(log_file_path)


def get_logger() -> Logger:
    return logging.getLogger()
