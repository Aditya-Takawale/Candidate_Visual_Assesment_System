"""
Structured Logger
Every log event carries: candidate_id, session_id, module, confidence, timestamp.
"""

from __future__ import annotations
import logging
import json
import sys
from datetime import datetime, timezone
from typing import Optional


class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": getattr(record, "cva_module", record.module),
            "candidate_id": getattr(record, "candidate_id", None),
            "session_id": getattr(record, "session_id", None),
            "confidence": getattr(record, "confidence", None),
            "msg": record.getMessage(),
        }
        return json.dumps({k: v for k, v in payload.items() if v is not None})


def get_logger(
    name: str,
    candidate_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

    class _Adapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            kwargs.setdefault("extra", {})
            if candidate_id:
                kwargs["extra"]["candidate_id"] = candidate_id
            if session_id:
                kwargs["extra"]["session_id"] = session_id
            return msg, kwargs

    return _Adapter(logger, {})
