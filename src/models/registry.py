"""The identity registry, mapping each member to a customer_id.

Nothing in the source data links a team member to a customer, so we declare the link here. It is
what makes authentication useful: recognising Taps only matters if it lets us look up Taps's
purchase history.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .. import config

REGISTRY_PATH: Path = config.DATA / "identity_registry.json"


def load_registry(path: Path | None = None) -> dict[str, str]:
    path = path or REGISTRY_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No identity registry at {path}. Create it once the merged dataset exists, "
            f"see build_registry()."
        )

    with path.open() as fh:
        registry: dict[str, str] = json.load(fh)

    extra = set(registry) - set(config.MEMBERS)
    if extra:
        raise ValueError(f"Registry contains non-members: {sorted(extra)}")

    missing = set(config.MEMBERS) - set(registry)
    if missing:
        raise ValueError(f"Registry is missing members: {sorted(missing)}")

    return registry


def save_registry(registry: dict[str, str], path: Path | None = None) -> None:
    path = path or REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(registry, fh, indent=2, sort_keys=True)
        fh.write("\n")


def build_registry(merged: pd.DataFrame, id_column: str = "customer_id") -> dict[str, str]:
    """Assign each member a distinct customer_id.

    Sorted order, not random, so the demo and the report's worked example stay reproducible.
    """
    if id_column not in merged.columns:
        raise ValueError(
            f"Merged dataset has no {id_column!r} column; got {list(merged.columns)[:10]}"
        )

    ids = sorted(merged[id_column].dropna().unique())
    if len(ids) < len(config.MEMBERS):
        raise ValueError(
            f"Merged dataset has only {len(ids)} distinct customers; need at least "
            f"{len(config.MEMBERS)}, one per member."
        )

    return {member: str(ids[i]) for i, member in enumerate(config.MEMBERS)}


def validate_registry(
    registry: dict[str, str],
    merged: pd.DataFrame,
    id_column: str = "customer_id",
) -> None:
    if len(set(registry.values())) != len(registry):
        raise ValueError("Two members are mapped to the same customer_id; the mapping must be 1:1")

    present = set(merged[id_column].astype(str))
    dangling = {m: cid for m, cid in registry.items() if cid not in present}
    if dangling:
        raise ValueError(
            f"Registry points at customer_ids absent from the merged dataset: {dangling}"
        )
