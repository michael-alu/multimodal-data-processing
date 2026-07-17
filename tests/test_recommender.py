"""Tests for the product recommendation model.

The merged dataset does not exist yet, so these build synthetic tables that satisfy the contract in
schemas.validate_merged and nothing more.
"""

import numpy as np
import pandas as pd
import pytest

from src import schemas
from src.models.recommender import (
    Recommendation,
    StubRecommender,
    TrainedRecommender,
)

PRODUCTS = ["Earbuds", "Shoes", "Coffee Maker", "Backpack"]


def _merged(n_customers=60, rows_per_customer=1, seed=0):
    """Synthetic merged data where age and channel genuinely predict the product."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_customers):
        product = PRODUCTS[i % len(PRODUCTS)]
        for _ in range(rows_per_customer):
            rows.append({
                "customer_id": f"C{i:03d}",
                "age": 20 + PRODUCTS.index(product) * 10 + rng.normal(0, 1.5),
                "spend": 100 + PRODUCTS.index(product) * 50 + rng.normal(0, 8),
                "channel": ["web", "app", "store", "web"][PRODUCTS.index(product)],
                "product": product,
            })
    return pd.DataFrame(rows)


# --- the stub ---


def test_stub_is_deterministic_and_flagged():
    stub = StubRecommender()
    assert stub.is_stub is True
    assert stub.recommend("C002") == stub.recommend("C002")
    assert isinstance(stub.recommend("C002"), Recommendation)


def test_trained_recommender_is_not_flagged_as_a_stub():
    assert TrainedRecommender().is_stub is False


# --- fitting and recommending ---


def test_recommends_a_known_product_for_a_known_customer():
    df = _merged()
    model = TrainedRecommender().fit(df)
    rec = model.recommend("C000")
    assert rec.product in PRODUCTS
    assert 0.0 <= rec.confidence <= 1.0


def test_recommendation_matches_the_customers_actual_product():
    """Features are separable by construction, so a fitted model should recover the product."""
    df = _merged()
    model = TrainedRecommender().fit(df)
    hits = 0
    for cid in df["customer_id"].unique():
        expected = df[df["customer_id"] == cid]["product"].iloc[-1]
        if model.recommend(cid).product == expected:
            hits += 1
    assert hits / df["customer_id"].nunique() > 0.9


def test_unknown_customer_raises_rather_than_guessing():
    model = TrainedRecommender().fit(_merged())
    with pytest.raises(ValueError, match="Unknown customer_id"):
        model.recommend("C999")


def test_recommend_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        TrainedRecommender().recommend("C000")


def test_categorical_columns_are_encoded_not_crashed():
    """The merge will carry text columns; a forest cannot eat them raw."""
    df = _merged()
    assert df["channel"].dtype == object
    model = TrainedRecommender().fit(df)
    assert any(c.startswith("channel_") for c in model.encoded_columns_)


def test_id_and_target_are_never_used_as_features():
    """Leaking the target into the features would score perfectly and mean nothing."""
    model = TrainedRecommender().fit(_merged())
    assert not any("product" in c for c in model.encoded_columns_)
    assert "customer_id" not in model.encoded_columns_


# --- evaluation ---


def test_cross_validate_reports_all_three_metrics():
    m = TrainedRecommender().cross_validate(_merged())
    assert 0.0 <= m.accuracy <= 1.0
    assert 0.0 <= m.f1_macro <= 1.0
    assert m.log_loss >= 0.0
    assert m.n_samples == 60


def test_cross_validate_learns_separable_data():
    m = TrainedRecommender().cross_validate(_merged())
    assert m.accuracy > 0.8


def test_repeat_customers_do_not_span_the_split():
    """A customer with several transactions must not appear in train and test at once."""
    from sklearn.model_selection import GroupKFold

    df = _merged(n_customers=20, rows_per_customer=3)
    groups = df["customer_id"].to_numpy()
    for train_idx, test_idx in GroupKFold(n_splits=5).split(df, df["product"], groups):
        assert not set(groups[train_idx]) & set(groups[test_idx])


# --- the contract ---


def test_missing_target_column_is_rejected():
    df = _merged().drop(columns=["product"])
    with pytest.raises(ValueError, match="required column 'product' is missing"):
        TrainedRecommender().fit(df)


def test_missing_id_column_is_rejected():
    df = _merged().drop(columns=["customer_id"])
    with pytest.raises(ValueError, match="required column 'customer_id' is missing"):
        TrainedRecommender().fit(df)


def test_null_target_is_rejected():
    df = _merged()
    df.loc[0, "product"] = None
    with pytest.raises(ValueError, match="null value"):
        TrainedRecommender().fit(df)


def test_single_product_is_rejected():
    df = _merged()
    df["product"] = "Earbuds"
    with pytest.raises(ValueError, match="at least 2 distinct products"):
        TrainedRecommender().fit(df)


def test_dataset_with_no_features_is_rejected():
    df = _merged()[["customer_id", "product"]]
    with pytest.raises(ValueError, match="no feature columns"):
        TrainedRecommender().fit(df)


# --- persistence ---


def test_save_load_round_trip(tmp_path):
    df = _merged()
    model = TrainedRecommender().fit(df)
    before = model.recommend("C001")

    path = tmp_path / "rec.joblib"
    model.save(path)
    after = TrainedRecommender.load(path).recommend("C001")

    assert before == after


def test_refuses_to_save_unfitted(tmp_path):
    with pytest.raises(RuntimeError, match="unfitted"):
        TrainedRecommender().save(tmp_path / "rec.joblib")
