"""The identity registry: the bridge between the biometric data and the tabular data.

Nothing in the source data links a team member to a customer. `customer_social_profiles` and
`customer_transactions` describe anonymous customers; the faces and voices are us. So the link is
something we declare, and this module is where it is declared.

The registry maps each member to exactly one `customer_id` in the merged dataset. That mapping is
what gives the authentication a point: recognising Taps is only useful if it lets us look up
*Taps's* purchase history and recommend a product for him specifically.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .. import config

REGISTRY_PATH = config.DATA / "identity_registry.json"


def load_registry(path: Path | None = None) -> dict[str, str]:
    """Return the member -> customer_id mapping."""
    path = path or REGISTRY_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No identity registry at {path}. Create it once the merged dataset exists — "
            f"see build_registry()."
        )
    with path.open() as fh:
        registry = json.load(fh)

    unknown = set(registry) - set(config.MEMBERS)
    if unknown:
        raise ValueError(f"Registry contains non-members: {sorted(unknown)}")
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
    """Assign each member a distinct customer_id from the merged dataset.

    Picks the first N distinct ids in sorted order so the mapping is deterministic and
    reproducible across runs — a random assignment would make the demo unrepeatable and the
    report's worked example wrong on the next run.
    """
    if id_column not in merged.columns:
        raise ValueError(f"Merged dataset has no {id_column!r} column; got {list(merged.columns)[:10]}")

    ids = sorted(merged[id_column].dropna().unique())
    if len(ids) < len(config.MEMBERS):
        raise ValueError(
            f"Merged dataset has only {len(ids)} distinct customers; need at least "
            f"{len(config.MEMBERS)}, one per member."
        )
    return {member: str(ids[i]) for i, member in enumerate(config.MEMBERS)}


def validate_registry(registry: dict[str, str], merged: pd.DataFrame, id_column: str = "customer_id") -> None:
    """Check the registry actually resolves against the merged dataset."""
    if len(set(registry.values())) != len(registry):
        raise ValueError("Two members are mapped to the same customer_id; the mapping must be 1:1")

    present = set(merged[id_column].astype(str))
    dangling = {m: cid for m, cid in registry.items() if cid not in present}
    if dangling:
        raise ValueError(f"Registry points at customer_ids absent from the merged dataset: {dangling}")
