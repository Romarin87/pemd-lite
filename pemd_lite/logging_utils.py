from __future__ import annotations

import io
import logging
import sys
from pathlib import Path


class _LoggerWriter(io.TextIOBase):
    def __init__(self, logger: logging.Logger, level: int):
        self._logger = logger
        self._level = level
        self._buffer = ""

    def write(self, data):
        if not data:
            return 0
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip()
            if line:
                self._logger.log(self._level, line)
        return len(data)

    def flush(self):
        line = self._buffer.rstrip()
        if line:
            self._logger.log(self._level, line)
        self._buffer = ""


def configure_workflow_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    if root.handlers:
        for handler in list(root.handlers):
            root.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.__stdout__)
    stream_handler.setFormatter(formatter)

    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    sys.stdout = _LoggerWriter(logging.getLogger("stdout"), logging.INFO)
    sys.stderr = _LoggerWriter(logging.getLogger("stderr"), logging.ERROR)

    return logging.getLogger("workflow")
