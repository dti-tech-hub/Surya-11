"""Shared utilities for PowerShield-FL."""

from __future__ import annotations

import sys

# Never let a legacy Windows console code page crash a run while printing a
# scientific symbol (arrows, plus/minus, etc.). Log files are written as UTF-8
# separately; only the console gets replacement characters when unsupported.
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass
