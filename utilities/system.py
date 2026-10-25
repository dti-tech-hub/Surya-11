"""Environment and resource inspection."""
from __future__ import annotations

import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil


def _version(module: str) -> str | None:
    try:
        mod = __import__(module)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return None


def environment_info(project_root: str | Path | None = None) -> dict[str, Any]:
    vitals = psutil.virtual_memory()
    cpu = platform.processor() or "unknown"
    try:
        cpu = platform.uname().processor or cpu
    except Exception:
        pass
    disk = shutil.disk_usage(project_root or Path.cwd())
    gpu = "not available"
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
        else:
            gpu = "CPU only"
    except Exception:
        pass
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": cpu,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(vitals.total / (1024**3), 2),
        "disk_free_gb": round(disk.free / (1024**3), 2),
        "gpu": gpu,
        "packages": {
            name: _version(name)
            for name in ["numpy", "pandas", "sklearn", "torch", "matplotlib", "scipy", "shap", "streamlit", "yaml"]
        },
    }


def check_dependencies() -> dict[str, bool]:
    import importlib.util

    required = ["numpy", "pandas", "sklearn", "torch", "matplotlib", "yaml", "scipy"]
    optional = ["shap", "streamlit", "plotly", "psutil"]
    return {
        **{m: importlib.util.find_spec(m) is not None for m in required},
        **{m: importlib.util.find_spec(m) is not None for m in optional},
    }


def environment_ok() -> tuple[bool, list[str]]:
    issues: list[str] = []
    if sys.version_info < (3, 10):
        issues.append(f"Python >= 3.10 required, found {sys.version.split()[0]}")
    for name, present in check_dependencies().items():
        if not present:
            issues.append(f"Missing dependency: {name}")
    if psutil.virtual_memory().available < 1.5 * (1024**3):
        issues.append("Less than 1.5 GB RAM available")
    return not issues, issues


def package_versions() -> dict[str, str | None]:
    return environment_info().get("packages", {})
