"""
run_logger.py -- Automatic run logging for Sbocker.py

Captures all stdout + stderr (including unhandled exception tracebacks) to a
timestamped log file, with API key scrubbing on write to the file (console stays
untouched). Installed as a tee at the start of main() via start_run_logging().

Key-scrubbing pattern reused from baseline_tools/pipeline_analysis.py to prevent
accidental key leaks to disk while keeping console output as-is.
"""

import sys
import os
import re
from datetime import datetime
from pathlib import Path


# Reuse the key-scrubbing pattern from baseline_tools/pipeline_analysis.py
_APIKEY_RE = re.compile(r"apikey=[^&\s\"']+", re.IGNORECASE)


def scrub(text, key=None):
    """
    Strip any apikey=... query-param AND any literal key substring from `text`.

    Public API for masking sensitive keys in log output.

    Args:
        text: the text to scrub
        key: optional api_key literal to also mask

    Returns:
        scrubbed text with apikey=*** and optional key literal masked
    """
    s = _APIKEY_RE.sub("apikey=***", str(text))
    if key:
        s = s.replace(str(key), "***")
    return s


# Keep internal alias for backward compatibility within the module
_scrub = scrub


class _TeeStream:
    """Write-through stream wrapper that duplicates writes to both the original
    stream (console) and a log file. Scrubs api_key from file writes only.
    Flushes on every write for crash-safety."""

    def __init__(self, original_stream, log_file, scrub_key=None):
        """
        Args:
            original_stream: sys.stdout or sys.stderr
            log_file: open file object to write to
            scrub_key: optional api_key to scrub from log file (not from console)
        """
        self._original = original_stream
        self._log_file = log_file
        self._scrub_key = scrub_key

    def write(self, text):
        # Console: write as-is, unmodified
        if self._original:
            self._original.write(text)
            self._original.flush()

        # Log file: write scrubbed (apikey masked)
        if self._log_file:
            scrubbed = _scrub(text, self._scrub_key)
            self._log_file.write(scrubbed)
            self._log_file.flush()

        return len(text)

    def flush(self):
        if self._original:
            self._original.flush()
        if self._log_file:
            self._log_file.flush()

    def isatty(self):
        return False


def start_run_logging(api_key=None):
    """
    Install a tee on sys.stdout and sys.stderr that duplicates all output to a
    timestamped log file in logs/ while preserving console output.

    Args:
        api_key: optional API key to scrub from the log file (console unaffected)

    Returns:
        tuple: (log_path_str, log_file_object)
            log_path_str: absolute path to the log file, for display/logging
            log_file_object: the open file object (caller should close it in cleanup)
    """

    # Create logs/ directory if absent
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Timestamped log filename: run_20260716_142530.log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"run_{timestamp}.log"
    log_path = logs_dir / log_filename

    # Open log file
    log_file = open(log_path, "w", buffering=1)  # line-buffered

    # Write header to log file (not to console)
    log_file.write(f"=== Run started at {datetime.now().isoformat()} ===\n")
    log_file.write(f"Log file: {log_path.absolute()}\n")
    log_file.flush()

    # Print absolute path to console at START
    print(f"\n[RUN LOG] Logging to: {log_path.absolute()}\n", flush=True)
    log_file.write(f"[RUN LOG] Logging to: {log_path.absolute()}\n")
    log_file.flush()

    # Replace sys.stdout and sys.stderr with tee wrappers
    sys.stdout = _TeeStream(sys.__stdout__, log_file, scrub_key=api_key)
    sys.stderr = _TeeStream(sys.__stderr__, log_file, scrub_key=api_key)

    return str(log_path.absolute()), log_file


def end_run_logging(log_path_str, log_file_obj):
    """
    Close the run log and restore sys.stdout/stderr.

    Args:
        log_path_str: absolute path returned from start_run_logging()
        log_file_obj: log file object returned from start_run_logging()
    """
    # Write footer to log file
    log_file_obj.write(f"\n=== Run ended at {datetime.now().isoformat()} ===\n")
    log_file_obj.flush()
    log_file_obj.close()

    # Restore original streams
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    # Print path to console at END
    print(f"\n[RUN LOG] Log file complete: {log_path_str}\n", flush=True)
