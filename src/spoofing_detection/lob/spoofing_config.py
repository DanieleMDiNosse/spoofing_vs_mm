from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPOOFING_CONFIG_PATH = REPO_ROOT / "configs" / "spoofing_detection_parameters.json"
_LIST_KEYS = {"depth_grid", "gamma_grid", "execution_anchor_modes"}
ACTOR_IDENTITY_MODES = ("client_then_firm",)
EXECUTION_ANCHOR_MODES = ("passive", "aggressive")


def _is_comment_key(value: Any) -> bool:
    return str(value).strip().lower().startswith("_comment_")


def validate_actor_identity_mode(value: Any) -> str:
    mode = str(value).strip().lower()
    if mode not in ACTOR_IDENTITY_MODES:
        raise ValueError(f"unknown actor identity mode: {mode or '<empty>'}")
    return mode


def parse_execution_anchor_modes(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        modes = [part.strip().lower() for part in value.split(",") if part.strip()]
    elif isinstance(value, Iterable) and not isinstance(value, Mapping):
        modes = [str(part).strip().lower() for part in value if str(part).strip()]
    else:
        modes = []
    if not modes:
        raise ValueError("execution anchor modes must contain at least one value")
    if len(modes) != len(set(modes)):
        raise ValueError("execution anchor modes must not contain duplicates")
    unknown = sorted(set(modes) - set(EXECUTION_ANCHOR_MODES))
    if unknown:
        raise ValueError(f"unknown execution anchor mode(s): {', '.join(unknown)}")
    return tuple(mode for mode in EXECUTION_ANCHOR_MODES if mode in modes)


def parse_msci_threshold_by_anchor(value: Any) -> dict[str, float | None]:
    """Parse explicit per-anchor thresholds; null marks an uncalibrated branch."""

    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, Mapping) or not parsed:
        raise ValueError("msci threshold by anchor must be a non-empty JSON object")
    normalized = {
        str(anchor).strip().lower(): threshold
        for anchor, threshold in parsed.items()
        if not _is_comment_key(anchor)
    }
    if not normalized:
        raise ValueError("msci threshold by anchor must be a non-empty JSON object")
    unknown = sorted(set(normalized) - set(EXECUTION_ANCHOR_MODES))
    if unknown:
        raise ValueError(f"unknown execution anchor mode(s): {', '.join(unknown)}")

    result: dict[str, float | None] = {}
    for anchor in EXECUTION_ANCHOR_MODES:
        if anchor not in normalized:
            continue
        threshold = normalized[anchor]
        if threshold is None:
            result[anchor] = None
            continue
        try:
            numeric_threshold = float(threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"MSCI threshold for {anchor} must be finite or null") from exc
        if not math.isfinite(numeric_threshold):
            raise ValueError(f"MSCI threshold for {anchor} must be finite or null")
        result[anchor] = numeric_threshold
    return result


def _normalise_config_key(key: str) -> str:
    return "lambda_" if key == "lambda" else key


def _normalise_config_value(key: str, value: Any) -> Any:
    if key in _LIST_KEYS and isinstance(value, list):
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
    new dependency. Adjacent documentation keys prefixed with ``_comment_`` are
    ignored. A top-level ``shared`` section is applied before the named section.
    CLI flags still override these defaults in each script.
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
            if _is_comment_key(raw_key):
                continue
            key = _normalise_config_key(str(raw_key))
            if key not in allowed:
                allowed_text = ", ".join(sorted(allowed))
                raise ValueError(
                    f"unknown key `{raw_key}` in spoofing config section `{section_name}`; "
                    f"allowed keys: {allowed_text}"
                )
            defaults[key] = _normalise_config_value(key, value)
    return defaults
