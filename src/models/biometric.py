"""Face recognition and voiceprint verification.

Both answer the same question, "which member is this", so they share one class and differ only in
which feature columns they read.

Every split is grouped by source_file. Augmentations of one photo are not independent samples, so
if some land in train and others in test the model is scored on recognising a photo it has already
seen, which reads near perfect and means nothing.
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
from .decision import ModalityResult


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    f1_macro: float
    log_loss: float
    n_samples: int
    n_folds: int

    def __str__(self) -> str:
        return (
            f"accuracy={self.accuracy:.3f}  f1_macro={self.f1_macro:.3f}  "
            f"log_loss={self.log_loss:.3f}  (n={self.n_samples}, {self.n_folds}-fold grouped CV)"
        )


class BiometricModel:
    """A member classifier over one modality's feature table."""

    def __init__(
        self,
        feature_columns: list[str],
        label_column: str = "member",
        group_column: str = "source_file",
        n_estimators: int = 300,
        random_state: int = 42,
    ) -> None:
        self.feature_columns: list[str] = feature_columns
        self.label_column: str = label_column
        self.group_column: str = group_column
        self.n_estimators: int = n_estimators
        self.random_state: int = random_state
        self.classes_: list[str] = []
        self._clf: RandomForestClassifier | None = None

    def _new_clf(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def _xy(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Feature table is missing {len(missing)} column(s), e.g. {missing[:5]}"
            )
        return df[self.feature_columns].to_numpy(), df[self.label_column].to_numpy()

    def fit(self, df: pd.DataFrame) -> "BiometricModel":
        X, y = self._xy(df)
        self._clf = self._new_clf()
        self._clf.fit(X, y)
        self.classes_ = list(self._clf.classes_)
        return self

    def predict(self, features: np.ndarray) -> ModalityResult:
        """Identify one sample. The caller applies the confidence floor, not this method."""
        if self._clf is None:
            raise RuntimeError("Model is not fitted; call fit() first")

        features = np.asarray(features).reshape(1, -1)
        if features.shape[1] != len(self.feature_columns):
            raise ValueError(
                f"Expected {len(self.feature_columns)} features, got {features.shape[1]}"
            )

        proba = self._clf.predict_proba(features)[0]
        best = int(np.argmax(proba))
        return ModalityResult(
            identity=str(self._clf.classes_[best]),
            confidence=float(proba[best]),
        )

    def cross_validate(self, df: pd.DataFrame, n_folds: int | None = None) -> Metrics:
        X, y = self._xy(df)
        groups = df[self.group_column].to_numpy()
        labels = sorted(set(y))

        n_groups = len(set(groups))
        if n_groups < 2:
            raise ValueError(
                f"Need at least 2 distinct {self.group_column} values to cross-validate; "
                f"got {n_groups}"
            )

        # The fold count is limited by the least represented member, not the total group count.
        # A member with 2 clips supports a 2-fold split and no more. Images give 3, audio 2.
        per_member = pd.Series(y).groupby(pd.Series(groups)).first().value_counts().min()
        if n_folds is None:
            n_folds = min(3, int(per_member))
        n_folds = max(2, min(n_folds, n_groups))

        y_true: list[str] = []
        y_pred: list[str] = []
        y_proba: list[np.ndarray] = []

        splitter = GroupKFold(n_splits=n_folds)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
            # A fold that trains without a member makes them unpredictable and the metrics junk.
            # Equal sized groups normally prevent this, but a dropped sample breaks that.
            train_labels = set(y[train_idx])
            if train_labels != set(labels):
                raise ValueError(
                    f"Fold {fold} trains without {sorted(set(labels) - train_labels)}: "
                    f"the {self.group_column} groups are too unbalanced for {n_folds}-fold CV. "
                    f"Check for dropped samples, or lower n_folds."
                )

            clf = self._new_clf().fit(X[train_idx], y[train_idx])
            fold_proba = clf.predict_proba(X[test_idx])

            # Align to the global label set so log_loss sees consistent classes.
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
            raise RuntimeError("Refusing to save an unfitted model")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "BiometricModel":
        return joblib.load(path)


def face_model() -> BiometricModel:
    """Recognise a member from their face. Trained on image_features.csv."""
    return BiometricModel(feature_columns=schemas.IMAGE_FEATURE_COLUMNS)


def voice_model() -> BiometricModel:
    """Verify a member from their voiceprint. Trained on audio_features.csv."""
    return BiometricModel(feature_columns=schemas.AUDIO_FEATURE_COLUMNS)
