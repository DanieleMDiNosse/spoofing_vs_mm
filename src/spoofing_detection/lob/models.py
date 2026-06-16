from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass
class ActiveOrder:
    order_id: str
    side: str
    price: float
    leaves_qty: float
    displayed_qty: float
    order_qty: float | None
    order_priority: str | None
    order_type_code: int | None
    order_type_label: str | None
    time_in_force_code: int | None
    firm_id: str | None
    client_original_id: str | None
    first_seen_sort_index: int
    last_update_sort_index: int
    last_event_class: str


@dataclass
class ReconstructionResult:
    panel: pl.DataFrame
    normalized_events: pl.DataFrame
    agent_event_state_panel: pl.DataFrame
    active_order_snapshots: pl.DataFrame
    price_level_depth_snapshots: pl.DataFrame
    validation: dict


@dataclass(frozen=True)
class OutputPaths:
    panel_path: Path
    normalized_path: Path
    agent_panel_path: Path
    active_orders_path: Path
    price_level_depth_path: Path
    metadata_path: Path
    validation_path: Path
