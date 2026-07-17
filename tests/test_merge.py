"""Tests for the tabular clean and merge.

Uses small handmade frames that reproduce the real quirks: string vs integer ids, one social row
per platform, several transactions per customer, and null ratings.
"""

import pandas as pd
import pytest

from src import schemas
from src.tabular import merge


def _social():
    return pd.DataFrame([
        {"customer_id_new": "A100", "social_media_platform": "Twitter",
         "engagement_score": 70, "purchase_interest_score": 4.0, "review_sentiment": "Positive"},
        {"customer_id_new": "A100", "social_media_platform": "Instagram",
         "engagement_score": 80, "purchase_interest_score": 4.4, "review_sentiment": "Negative"},
        {"customer_id_new": "A101", "social_media_platform": "LinkedIn",
         "engagement_score": 60, "purchase_interest_score": 2.0, "review_sentiment": "Neutral"},
        {"customer_id_new": "A999", "social_media_platform": "TikTok",
         "engagement_score": 50, "purchase_interest_score": 1.0, "review_sentiment": "Positive"},
    ])


def _transactions():
    return pd.DataFrame([
        {"customer_id_legacy": 100, "transaction_id": 1, "purchase_amount": 400,
         "purchase_date": "2024-01-01", "product_category": "Books", "customer_rating": 4.0},
        {"customer_id_legacy": 100, "transaction_id": 2, "purchase_amount": 200,
         "purchase_date": "2024-01-05", "product_category": "Books", "customer_rating": None},
        {"customer_id_legacy": 100, "transaction_id": 3, "purchase_amount": 300,
         "purchase_date": "2024-02-01", "product_category": "Sports", "customer_rating": 3.0},
        {"customer_id_legacy": 101, "transaction_id": 4, "purchase_amount": 250,
         "purchase_date": "2024-01-02", "product_category": "Electronics", "customer_rating": 2.0},
        {"customer_id_legacy": 555, "transaction_id": 5, "purchase_amount": 999,
         "purchase_date": "2024-03-01", "product_category": "Clothing", "customer_rating": 5.0},
    ])


# --- the join key ---


def test_string_and_integer_ids_are_reconciled():
    merged = merge.build_merged(_social(), _transactions())
    assert set(merged["customer_id"]) == {"100", "101"}


def test_only_customers_in_both_sources_survive():
    """A999 has no transactions, 555 has no profile; neither should appear."""
    merged = merge.build_merged(_social(), _transactions())
    assert "999" not in set(merged["customer_id"])
    assert "555" not in set(merged["customer_id"])


def test_malformed_social_id_is_rejected():
    bad = _social()
    bad.loc[0, "customer_id_new"] = "XYZ"
    with pytest.raises(ValueError, match="do not look like"):
        merge.clean_social(bad)


# --- the fan-out ---


def test_output_is_one_row_per_customer():
    merged = merge.build_merged(_social(), _transactions())
    assert len(merged) == merged["customer_id"].nunique()


def test_no_cross_product_blowup():
    """C100 has 2 social rows and 3 transactions. A raw join makes 6 rows; we want 1."""
    merged = merge.build_merged(_social(), _transactions())
    assert (merged["customer_id"] == "100").sum() == 1


def test_dominant_category_becomes_the_target():
    """C100 bought Books twice and Sports once, so the target is Books."""
    merged = merge.build_merged(_social(), _transactions())
    row = merged[merged["customer_id"] == "100"].iloc[0]
    assert row["product_category"] == "Books"


# --- cleaning ---


def test_exact_duplicate_social_rows_are_dropped():
    dup = pd.concat([_social(), _social().iloc[[0]]], ignore_index=True)
    cleaned = merge.clean_social(dup)
    assert (cleaned["customer_id"] == 100).sum() == 2  # 2 platforms, the exact dup is gone


def test_null_rating_is_filled_not_dropped():
    """C100 has a null rating on one transaction; the customer must still appear."""
    merged = merge.build_merged(_social(), _transactions())
    assert "100" in set(merged["customer_id"])
    assert merged["mean_customer_rating"].notna().all()


def test_platform_count_is_preserved_as_a_feature():
    merged = merge.build_merged(_social(), _transactions())
    row = merged[merged["customer_id"] == "100"].iloc[0]
    assert row["n_platforms"] == 2


# --- the contract and end to end ---


def test_merged_output_satisfies_the_model_contract():
    merged = merge.build_merged(_social(), _transactions())
    schemas.validate_merged(merged)  # must not raise


def test_no_nulls_survive():
    merged = merge.build_merged(_social(), _transactions())
    assert merged.isna().sum().sum() == 0


def test_merge_is_deterministic():
    a = merge.build_merged(_social(), _transactions())
    b = merge.build_merged(_social(), _transactions())
    pd.testing.assert_frame_equal(a, b)


def test_main_writes_a_file(tmp_path):
    social = tmp_path / "s.csv"
    transactions = tmp_path / "t.csv"
    out = tmp_path / "merged.csv"
    _social().to_csv(social, index=False)
    _transactions().to_csv(transactions, index=False)

    code = merge.main(["--social", str(social), "--transactions", str(transactions),
                       "--out", str(out)])
    assert code == 0
    assert out.exists()
    schemas.validate_merged(pd.read_csv(out))
