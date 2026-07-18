"""Product recommendation, the third model.

Predicts which product a customer buys, from the merged tabular dataset. The gate resolves a face
and voice to a customer_id, and this turns that id into a recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold

from .. import schemas
from .biometric import Metrics


@dataclass(frozen=True)
class Recommendation:
    product: str
    confidence: float

    def __str__(self) -> str:
        return f"{self.product} ({self.confidence:.0%} confidence)"


class Recommender:
    """Base class. Subclasses predict the product a customer is most likely to buy."""

    is_stub: bool = False

    def recommend(self, customer_id: str) -> Recommendation:
        raise NotImplementedError


class StubRecommender(Recommender):
    """Placeholder until the merged dataset lands. Announces itself so it cannot reach the demo."""

    is_stub = True

    PRODUCTS: list[str] = ["Wireless Earbuds", "Running Shoes", "Coffee Maker", "Backpack"]

    def recommend(self, customer_id: str) -> Recommendation:
        index = sum(ord(c) for c in str(customer_id)) % len(self.PRODUCTS)
        return Recommendation(product=self.PRODUCTS[index], confidence=0.0)


class TrainedRecommender(Recommender):
    """Random forest over the merged dataset.

    At demo time we only know a customer_id, not their features, so the fitted model keeps the
    customer rows and looks them up on recommend().
    """

    def __init__(
        self,
        target_column: str = schemas.MERGED_TARGET_COLUMN,
        id_column: str = schemas.MERGED_ID_COLUMN,
        n_estimators: int = 300,
        random_state: int = 42,
    ) -> None:
        self.target_column: str = target_column
        self.id_column: str = id_column
        self.n_estimators: int = n_estimators
        self.random_state: int = random_state
        self.classes_: list[str] = []
        self.encoded_columns_: list[str] = []
        self._clf: RandomForestClassifier | None = None
        self._customers: pd.DataFrame | None = None

    def _new_clf(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def _encode(self, df: pd.DataFrame, align: bool = True) -> pd.DataFrame:
        """One-hot the categorical columns. The merged data carries text fields a forest cannot eat."""
        features = df.drop(columns=[self.target_column, self.id_column], errors="ignore")
        encoded = pd.get_dummies(features)
        if align and self.encoded_columns_:
            encoded = encoded.reindex(columns=self.encoded_columns_, fill_value=0)
        return encoded

    def fit(self, merged: pd.DataFrame) -> "TrainedRecommender":
        schemas.validate_merged(merged)

        encoded = self._encode(merged, align=False)
        self.encoded_columns_ = list(encoded.columns)

        self._clf = self._new_clf()
        self._clf.fit(encoded.to_numpy(), merged[self.target_column].to_numpy())
        self.classes_ = list(self._clf.classes_)

        # Keep one row per customer for lookup. Last wins if a customer has several rows.
        customers = merged.drop_duplicates(subset=[self.id_column], keep="last").copy()
        customers.index = customers[self.id_column].astype(str)
        self._customers = customers
        return self

    def recommend(self, customer_id: str) -> Recommendation:
        if self._clf is None or self._customers is None:
            raise RuntimeError("Recommender is not fitted; call fit() first")

        key = str(customer_id)
        if key not in self._customers.index:
            raise ValueError(f"Unknown customer_id {key!r}, not present in the merged dataset")

        row = self._customers.loc[[key]]
        proba = self._clf.predict_proba(self._encode(row).to_numpy())[0]
        best = int(np.argmax(proba))
        return Recommendation(
            product=str(self._clf.classes_[best]),
            confidence=float(proba[best]),
        )

    def cross_validate(self, merged: pd.DataFrame, n_folds: int = 5) -> Metrics:
        """Grouped by customer, so a customer with several transactions cannot span the split."""
        schemas.validate_merged(merged)

        encoded = self._encode(merged, align=False)
        self.encoded_columns_ = list(encoded.columns)

        X = encoded.to_numpy()
        y = merged[self.target_column].to_numpy()
        groups = merged[self.id_column].astype(str).to_numpy()
        labels = sorted(set(y))

        n_groups = len(set(groups))
        if n_groups < 2:
            raise ValueError(f"Need at least 2 distinct customers to cross-validate; got {n_groups}")
        n_folds = max(2, min(n_folds, n_groups))

        y_true: list[str] = []
        y_pred: list[str] = []
        y_proba: list[np.ndarray] = []

        for train_idx, test_idx in GroupKFold(n_splits=n_folds).split(X, y, groups):
            clf = self._new_clf().fit(X[train_idx], y[train_idx])
            fold_proba = clf.predict_proba(X[test_idx])

            proba = np.zeros((len(test_idx), len(labels)))
            for j, cls in enumerate(clf.classes_):
                proba[:, labels.index(cls)] = fold_proba[:, j]

            y_true.extend(y[test_idx])
            y_pred.extend(clf.predict(X[test_idx]))
            y_proba.extend(proba)

        return Metrics(
            accuracy=float(accuracy_score(y_true, y_pred)),
            f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            log_loss=float(log_loss(y_true, np.array(y_proba), labels=labels)),
            n_samples=len(y_true),
            n_folds=n_folds,
        )

    def save(self, path: Path) -> None:
        if self._clf is None:
            raise RuntimeError("Refusing to save an unfitted recommender")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "TrainedRecommender":
        return joblib.load(path)
