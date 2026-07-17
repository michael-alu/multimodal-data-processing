"""Face recognition and voiceprint verification.

Both tasks are the same shape — given a feature vector, say which member this is and how
confident we are — so they share one implementation, differing only in which columns they read
and how many groups the cross-validation has.

On honest evaluation
--------------------
The augmentations are not independent samples. Six augmented copies of `taps_neutral.jpg` all
derive from one photo, and if some land in train while others land in test, the model is scored on
recognising a *photograph it has already seen*, not on recognising Taps. That inflates accuracy to
near-perfect and tells us nothing.

So every split here is grouped by `source_file`: all rows derived from one recording or photo stay
on the same side of the split. With 3 photos per member, a 3-fold grouped CV holds out one whole
expression per member per fold — the model must recognise Taps smiling having only ever trained on
Taps neutral and surprised. That is the question we actually care about, and the resulting numbers
are lower and real.
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
    """Cross-validated performance. Reported per model for rubric criterion 8."""

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
    ):
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.group_column = group_column
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.classes_: list[str] = []
        self._clf: RandomForestClassifier | None = None

    def _new_clf(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            # The dataset is tiny and perfectly balanced by construction; leaving trees
            # unpruned is fine and keeps probability estimates from collapsing to 0/1.
            min_samples_leaf=1,
            n_jobs=-1,
        )

    def _xy(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Feature table is missing {len(missing)} column(s), e.g. {missing[:5]}")
        return df[self.feature_columns].to_numpy(), df[self.label_column].to_numpy()

    def fit(self, df: pd.DataFrame) -> "BiometricModel":
        X, y = self._xy(df)
        self._clf = self._new_clf()
        self._clf.fit(X, y)
        self.classes_ = list(self._clf.classes_)
        return self

    def predict(self, features: np.ndarray) -> ModalityResult:
        """Identify one sample. Confidence is the winning class probability.

        The caller (`decision.authenticate`) applies the confidence floor that turns a weak
        match into a rejection — this method never decides access, it only reports a verdict.
        """
        if self._clf is None:
            raise RuntimeError("Model is not fitted; call fit() first")

        features = np.asarray(features).reshape(1, -1)
        if features.shape[1] != len(self.feature_columns):
            raise ValueError(
                f"Expected {len(self.feature_columns)} features, got {features.shape[1]}"
            )

        proba = self._clf.predict_proba(features)[0]
        idx = int(np.argmax(proba))
        return ModalityResult(identity=str(self._clf.classes_[idx]), confidence=float(proba[idx]))

    def cross_validate(self, df: pd.DataFrame, n_folds: int | None = None) -> Metrics:
        """Grouped cross-validation — augmentations of one source never span the split."""
        X, y = self._xy(df)
        groups = df[self.group_column].to_numpy()
        labels = sorted(set(y))

        n_groups = len(set(groups))
        if n_folds is None:
            # One fold per distinct source per member, capped by what the data supports.
            n_folds = min(3, n_groups)
        if n_groups < 2:
            raise ValueError(
                f"Need at least 2 distinct {self.group_column} values to cross-validate; got {n_groups}"
            )
        n_folds = min(n_folds, n_groups)

        y_true, y_pred, y_proba = [], [], []
        for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, groups)):
            # Our groups are equal-sized, so folds come out label-balanced. That stops being true
            # the moment a source drops out (e.g. a photo where face detection fails), and a fold
            # whose training set is missing a member makes that member unpredictable and the
            # metrics meaningless. Fail loudly rather than report a quietly wrong number.
            train_labels = set(y[train_idx])
            if train_labels != set(labels):
                raise ValueError(
                    f"Fold {fold} trains without {sorted(set(labels) - train_labels)}: "
                    f"the {self.group_column} groups are too unbalanced for {n_folds}-fold CV. "
                    f"Check for dropped samples, or lower n_folds."
                )

            clf = self._new_clf().fit(X[train_idx], y[train_idx])
            fold_proba = clf.predict_proba(X[test_idx])

            # Align columns to the global label set so log_loss sees consistent classes.
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


# --- the two concrete models --------------------------------------------


def face_model() -> BiometricModel:
    """Recognise a member from their face. Trained on Taps's image_features.csv."""
    return BiometricModel(feature_columns=schemas.IMAGE_FEATURE_COLUMNS)


def voice_model() -> BiometricModel:
    """Verify a member from their voiceprint. Trained on Tedla's audio_features.csv."""
    return BiometricModel(feature_columns=schemas.AUDIO_FEATURE_COLUMNS)
