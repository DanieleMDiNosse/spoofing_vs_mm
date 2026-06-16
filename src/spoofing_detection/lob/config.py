from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class LOBConfig:
    """Configuration for first-version visible LOB reconstruction.

    The defaults encode the user-approved decisions in
    docs/lob_reconstruction_plan.md: all events are included, top-N is 10,
    prices/quantities are used exactly as stored in parquet/CSV inputs, and
    the two first-version agent dimensions are FIRMID and
    NMSC_ORIGINALCLIENTIDSHORTCODE separately.
    """

    top_n: int = 10
    agent_dimensions: tuple[str, str] = ("firm", "client_original")
    include_all_events: bool = True
    strict_enums: bool = True
    use_parquet_values_as_is: bool = True
    snapshot_mode: str = "end_of_partition"

    def __post_init__(self) -> None:
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        if tuple(self.agent_dimensions) != ("firm", "client_original"):
            raise ValueError(
                "first-version implementation requires agent_dimensions "
                "('firm', 'client_original')"
            )
        if not self.include_all_events:
            raise ValueError("first-version implementation must include all events")
        if not self.use_parquet_values_as_is:
            raise ValueError("first-version implementation must use parquet values as-is")
        allowed_snapshot_modes = {"none", "every_event_for_sample", "issue_rows_only", "end_of_partition"}
        if self.snapshot_mode not in allowed_snapshot_modes:
            raise ValueError(
                f"snapshot_mode must be one of {sorted(allowed_snapshot_modes)}, got {self.snapshot_mode!r}"
            )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["agent_dimensions"] = list(self.agent_dimensions)
        return data
