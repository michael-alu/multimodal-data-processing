"""Clean and merge the two customer datasets into one row per customer.

    python -m src.tabular.merge

The join
--------
The sources share no key. Social profiles use customer_id_new ("A178"), transactions use
customer_id_legacy (178). Both number 100 to 199 and 61 customers overlap, so we strip the "A"
prefix and match on the integer. That is the only available link.

The fan-out
-----------
Both sides hold several rows per customer: social has one row per platform, transactions one row
per purchase. Joining them raw gives a partial cross product, 219 rows out of 117 real
transactions. So each side is aggregated to one row per customer first, then joined one to one.

One row per customer is also what the app needs. After a face and voice authenticate someone, the
only thing we know is their customer_id, so the model has to answer "what would this customer buy"
from their profile and history, not from a transaction that has not happened.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .. import config, schemas

SOCIAL_CSV: Path = config.RAW_TABULAR / "customer_social_profiles.csv"
TRANSACTIONS_CSV: Path = config.RAW_TABULAR / "customer_transactions.csv"


def _mode(series: pd.Series) -> str | float:
    """Most common value. Ties break alphabetically so the merge is reproducible."""
    counts = series.value_counts()
    if counts.empty:
        return float("nan")
    top = counts.max()
    return sorted(counts[counts == top].index)[0]


def clean_social(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.copy()
    df["customer_id"] = df["customer_id_new"].str.extract(r"^A(\d+)$")[0]

    unparsed = df["customer_id"].isna().sum()
    if unparsed:
        raise ValueError(f"{unparsed} social id(s) do not look like 'A123'")

    df["customer_id"] = df["customer_id"].astype(int)
    return df.drop(columns=["customer_id_new"])


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    df["customer_id"] = df["customer_id_legacy"].astype(int)
    df["purchase_date"] = pd.to_datetime(df["purchase_date"])
    return df.drop(columns=["customer_id_legacy"])


def aggregate_social(df: pd.DataFrame) -> pd.DataFrame:
    """One row per customer. Several rows means several platforms, not a duplicate."""
    return (
        df.groupby("customer_id")
        .agg(
            engagement_score=("engagement_score", "mean"),
            purchase_interest_score=("purchase_interest_score", "mean"),
            n_platforms=("social_media_platform", "nunique"),
            primary_platform=("social_media_platform", _mode),
            review_sentiment=("review_sentiment", _mode),
        )
        .reset_index()
    )


def aggregate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """One row per customer, with their dominant category as the target."""
    agg = (
        df.groupby("customer_id")
        .agg(
            n_transactions=("transaction_id", "count"),
            total_spend=("purchase_amount", "sum"),
            mean_purchase_amount=("purchase_amount", "mean"),
            mean_customer_rating=("customer_rating", "mean"),
            product_category=("product_category", _mode),
        )
        .reset_index()
    )

    # customer_rating has nulls, and a customer whose every rating is null aggregates to NaN.
    # Fill with the global mean so the row stays usable rather than being dropped.
    agg["mean_customer_rating"] = agg["mean_customer_rating"].fillna(
        df["customer_rating"].mean()
    )
    return agg


def build_merged(social: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Inner join on the canonical customer_id. Inner because a row needs both halves to be useful."""
    s = aggregate_social(clean_social(social))
    t = aggregate_transactions(clean_transactions(transactions))

    merged = t.merge(s, on="customer_id", how="inner", validate="one_to_one")
    merged["customer_id"] = merged["customer_id"].astype(str)
    return merged.sort_values("customer_id").reset_index(drop=True)


def report(social: pd.DataFrame, transactions: pd.DataFrame, merged: pd.DataFrame) -> None:
    """Post-merge validation, printed so it can go straight into the report."""
    s_ids = set(clean_social(social)["customer_id"])
    t_ids = set(clean_transactions(transactions)["customer_id"])

    print("=== inputs")
    print(f"  social profiles     : {len(social):>4} rows, {len(s_ids):>3} customers")
    print(f"  transactions        : {len(transactions):>4} rows, {len(t_ids):>3} customers")
    print(f"  exact duplicate rows dropped from social: {social.duplicated().sum()}")

    print("\n=== join")
    print(f"  key                 : strip 'A' from customer_id_new, match customer_id_legacy")
    print(f"  customers in both   : {len(s_ids & t_ids):>3}")
    print(f"  social only, dropped: {len(s_ids - t_ids):>3}")
    print(f"  transactions only, dropped: {len(t_ids - s_ids):>3}")

    print("\n=== fan-out avoided")
    naive = clean_transactions(transactions).merge(
        clean_social(social), on="customer_id", how="inner"
    )
    print(f"  naive row-level join: {len(naive):>4} rows")
    print(f"  after aggregating   : {len(merged):>4} rows, one per customer")

    print("\n=== output")
    print(f"  rows                : {len(merged)}")
    print(f"  unique customers    : {merged['customer_id'].nunique()}")
    print(f"  nulls               : {int(merged.isna().sum().sum())}")
    print(f"  target classes      : {merged[schemas.MERGED_TARGET_COLUMN].nunique()}")
    print(f"  target distribution : {dict(merged[schemas.MERGED_TARGET_COLUMN].value_counts())}")

    assert len(merged) == merged["customer_id"].nunique(), "one row per customer"
    assert merged.isna().sum().sum() == 0, "no nulls survive the merge"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean and merge the customer datasets")
    parser.add_argument("--social", type=Path, default=SOCIAL_CSV)
    parser.add_argument("--transactions", type=Path, default=TRANSACTIONS_CSV)
    parser.add_argument("--out", type=Path, default=config.MERGED_CSV)
    args = parser.parse_args(argv)

    social = pd.read_csv(args.social)
    transactions = pd.read_csv(args.transactions)

    merged = build_merged(social, transactions)
    schemas.validate_merged(merged)
    report(social, transactions, merged)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"\nsaved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
