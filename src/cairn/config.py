"""Load configuration from config.toml.

config.toml is optional — all settings have built-in defaults.
Partial files are fine; only the keys present are applied.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

# Absolute path to config.toml at project root
_CONFIG_PATH = Path(__file__).parent.parent / "config.toml"

_DEFAULTS = {
    "classifier": "claude-haiku-4-5-20251001",
    "evaluator": "claude-sonnet-4-5-20250929",
    "responder": "claude-opus-4-6",
    "renderer": "claude-sonnet-4-5-20250929",
}


def load_model_config() -> dict[str, str]:
    """Return model names for each pipeline stage.

    Reads [models] section from config.toml if the file exists.
    Falls back to defaults for any missing keys.
    """
    cfg = dict(_DEFAULTS)

    if _CONFIG_PATH.exists():
        with _CONFIG_PATH.open("rb") as f:
            data = tomllib.load(f)
        cfg.update(data.get("models", {}))

    return cfg
