import logging


def init_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
