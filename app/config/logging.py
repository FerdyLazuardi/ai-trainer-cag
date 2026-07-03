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
    # Try the bind-mounted ./logs first (host-visible, what devs expect on
    # local docker-compose). If the destination file is NOT writable —
    # which happens when the container runs as a non-root user and the
    # host mount is owned by a different UID, or when an old log file
    # was created by root and is mode 0644 — fall back to a per-user
    # cache path inside the container's writable layer. The console sink
    # above is always available, so logs are never lost.
    import os
    primary_log_dir = Path("logs")
    primary_log_path = primary_log_dir / "app.json"
    try:
        primary_log_dir.mkdir(exist_ok=True)
        # Must open the actual destination in append mode — a fresh-file
        # test misses the case where an existing file in the dir is owned
        # by a different user.
        with open(primary_log_path, "a"):
            pass
        structured_log_path = primary_log_path
    except (PermissionError, OSError):
        fallback_log_dir = Path(os.environ.get("XDG_CACHE_HOME", "/tmp")) / "cag-lms-agent-logs"
        fallback_log_dir.mkdir(parents=True, exist_ok=True)
        structured_log_path = fallback_log_dir / "app.json"
        logger.warning(
            "Primary log dir not writable, using fallback",
            primary=str(primary_log_path),
            fallback=str(structured_log_path),
        )

    logger.add(
        structured_log_path,
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
