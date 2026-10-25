"""Configuration loading with mode presets and dotted overrides."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
CONFIG_FILES = [
    "config.yaml",
    "datasets.yaml",
    "clients.yaml",
    "privacy.yaml",
    "aggregation.yaml",
    "experiments.yaml",
]


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file {path} must contain a mapping")
    return data


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` without mutating inputs."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def load_dotenv(path: Path | None = None) -> None:
    """Load a minimal .env file if present (never overrides real env vars)."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_config(
    mode: str | None = None,
    overrides: Iterable[str] | None = None,
    config_dir: str | Path | None = None,
) -> dict:
    """Load all configuration files, apply a mode preset and CLI overrides.

    Overrides use ``dotted.key=value`` syntax and are parsed as YAML values.
    """
    directory = Path(config_dir) if config_dir else CONFIG_DIR
    cfg: dict = {}
    for filename in CONFIG_FILES:
        cfg = deep_merge(cfg, _read_yaml(directory / filename))

    selected_mode = mode or get(cfg, "experiment.mode", "quick")
    presets = get(cfg, "modes", {}) or {}
    if selected_mode not in presets:
        raise ValueError(f"Unknown mode '{selected_mode}'. Available: {sorted(presets)}")
    for dotted, value in presets[selected_mode].items():
        set_dotted(cfg, dotted, value)
    cfg.setdefault("experiment", {})["mode"] = selected_mode

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must use dotted.key=value syntax")
        dotted, raw_value = item.split("=", 1)
        set_dotted(cfg, dotted.strip(), yaml.safe_load(raw_value))

    cfg["_meta"] = {
        "root": str(ROOT),
        "config_dir": str(directory),
        "mode": selected_mode,
        "python": os.sys.version.split()[0],
    }
    return cfg


def resolve_path(cfg: dict, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path
