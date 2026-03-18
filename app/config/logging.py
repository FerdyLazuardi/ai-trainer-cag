"""
Loguru structured logging configuration.
Sets up console + JSON file sinks with contextual fields.
"""
import sys
from pathlib import Path

from loguru import logger


def setup_logging(debug: bool = False) -> None:
    """Configure loguru sinks for the application."""
    logger.remove()

    log_level = "DEBUG" if debug else "INFO"

    # ─── Console sink (human-readable) ──────────────────────────────────────
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # ─── JSON file sink (structured, for production log aggregation) ─────────
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger.add(
        log_dir / "app.json",
        level="INFO",
        format="{message}",
        serialize=True,          # outputs as JSON
        rotation="50 MB",
        retention="14 days",
        compression="gz",
        enqueue=True,             # non-blocking async-safe
    )

    logger.info("Logging configured", level=log_level, debug=debug)


__all__ = ["setup_logging", "logger"]
