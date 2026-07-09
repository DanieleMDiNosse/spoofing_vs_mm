from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPOOFING_CONFIG_PATH = REPO_ROOT / "configs" / "spoofing_detection_parameters.json"
_GRID_KEYS = {"depth_grid", "gamma_grid"}


def _normalise_config_key(key: str) -> str:
    return "lambda_" if key == "lambda" else key


def _normalise_config_value(key: str, value: Any) -> Any:
    if key in _GRID_KEYS and isinstance(value, list):
        return ",".join(str(item) for item in value)
    return value


def load_spoofing_config_defaults(
    *,
    config_path: Path | None,
    section: str,
    allowed_keys: Iterable[str],
) -> dict[str, Any]:
    """Load CLI defaults for one spoofing pipeline section from JSON config.

    The config file is intentionally JSON-only so the repository does not need a
    new dependency. A top-level ``shared`` section is applied before the named
    section. CLI flags still override these defaults in each script.
    """

    path = config_path or DEFAULT_SPOOFING_CONFIG_PATH
    if not path.exists():
        return {}

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"spoofing config must be a JSON object: {path}")

    allowed = set(allowed_keys)
    defaults: dict[str, Any] = {}
    for section_name in ("shared", section):
        raw_section = payload.get(section_name, {})
        if raw_section is None:
            continue
        if not isinstance(raw_section, Mapping):
            raise ValueError(f"spoofing config section `{section_name}` must be an object: {path}")
        for raw_key, value in raw_section.items():
            key = _normalise_config_key(str(raw_key))
            if key not in allowed:
                allowed_text = ", ".join(sorted(allowed))
                raise ValueError(
                    f"unknown key `{raw_key}` in spoofing config section `{section_name}`; "
                    f"allowed keys: {allowed_text}"
                )
            defaults[key] = _normalise_config_value(key, value)
    return defaults
