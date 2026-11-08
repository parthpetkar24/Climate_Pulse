"""
================================================================================
  ClimatePulse Backend  —  Logging Configuration
================================================================================
"""
import logging
import sys

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Quiet down noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("multipart").setLevel(logging.WARNING)

logger = logging.getLogger("climatepulse.backend")

