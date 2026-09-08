from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Callable, Sequence

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
    return key


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
    """Load all scientific parameters for one spoofing pipeline section.

    The config file is intentionally JSON-only so the repository does not need a
    new dependency. Adjacent documentation keys prefixed with ``_comment_`` are
    ignored. The requested named section must exist and supply every allowed
    key: cross-section inheritance and silent hard-coded fallbacks are
    deliberately disabled.
    """

    path = config_path or DEFAULT_SPOOFING_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"spoofing configuration file not found: {path}. "
            "Scientific parameters must be supplied by one JSON configuration file."
        )

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"spoofing config must be a JSON object: {path}")

    raw_section = payload.get(section)
    if raw_section is None:
        raise ValueError(f"missing required section `{section}` in spoofing config: {path}")
    if not isinstance(raw_section, Mapping):
        raise ValueError(f"spoofing config section `{section}` must be an object: {path}")

    allowed = set(allowed_keys)
    defaults: dict[str, Any] = {}
    for raw_key, value in raw_section.items():
        if _is_comment_key(raw_key):
            continue
        key = _normalise_config_key(str(raw_key))
        if key not in allowed:
            allowed_text = ", ".join(sorted(allowed))
            raise ValueError(
                f"unknown key `{raw_key}` in spoofing config section `{section}`; "
                f"allowed keys: {allowed_text}"
            )
        defaults[key] = _normalise_config_value(key, value)

    missing = sorted(allowed - set(defaults))
    if missing:
        raise ValueError(
            f"missing required keys in spoofing config section `{section}`: {missing}. "
            "Hard-coded scientific defaults are disabled."
        )
    return defaults


def assert_config_parameters_match(
    *,
    config_path: Path,
    section: str,
    allowed_keys: Iterable[str],
    effective_parameters: Mapping[str, Any],
    normalizers: Mapping[str, Callable[[Any], Any]] | None = None,
) -> None:
    """Prevent JSON provenance from being attached to mismatched effective values."""

    configured = load_spoofing_config_defaults(
        config_path=config_path,
        section=section,
        allowed_keys=allowed_keys,
    )
    normalizers = normalizers or {}
    mismatched: list[str] = []
    for key, configured_value in configured.items():
        normalizer = normalizers.get(key, lambda value: value)
        if key not in effective_parameters or normalizer(configured_value) != normalizer(
            effective_parameters[key]
        ):
            mismatched.append(key)
    if mismatched:
        raise ValueError(
            f"effective parameters do not match JSON config section `{section}`: "
            f"{', '.join(sorted(mismatched))}"
        )


def reject_parameter_overrides(
    parser: Any,
    argv: Sequence[str] | None,
    *,
    parameter_options: Iterable[str],
) -> None:
    """Reject CLI options whose values must come from the JSON configuration."""

    tokens = list(sys.argv[1:] if argv is None else argv)
    explicit = sorted(
        option
        for option in set(parameter_options)
        if any(token == option or token.startswith(f"{option}=") for token in tokens)
    )
    if explicit:
        parser.error(
            "scientific parameter overrides are disabled; edit the selected JSON config "
            f"instead: {', '.join(explicit)}"
        )


def spoofing_config_provenance(
    config_path: Path | None,
    *,
    section: str,
) -> dict[str, str]:
    """Return auditable provenance for the authoritative JSON parameter source."""

    path = (config_path or DEFAULT_SPOOFING_CONFIG_PATH).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "parameter_source": "json_config_only",
        "config_section": section,
        "config": str(path),
        "config_sha256": digest,
    }
