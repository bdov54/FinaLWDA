import logging
import sys

def get_logger(name: str = "review_analytics") -> logging.Logger:
    lg = logging.getLogger(name)
    if lg.handlers:
        return lg

    lg.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    lg.addHandler(handler)
    lg.propagate = False
    return lg

logger = get_logger()