"""Exploratory data analysis on the merged customer dataset.

    python -m src.tabular.eda

Prints summary statistics and saves three labelled plots to reports/: the target class balance,
the spread and outliers of the numeric features, and a correlation heatmap. Each plot is described
in the printout so the interpretation goes straight into the report.

This is a starting point for Anthony to lift into the notebook and rewrite. Plain pandas and
matplotlib on purpose, nothing fancy to unpick.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save files, do not open a window
import matplotlib.pyplot as plt
import pandas as pd

from .. import config, schemas

# customer_id is an identifier, not a feature, so it is left out of the analysis.
ID_COLUMN = schemas.MERGED_ID_COLUMN
TARGET = schemas.MERGED_TARGET_COLUMN


def numeric_columns(df: pd.DataFrame) -> list[str]:
    cols = df.select_dtypes(include="number").columns.tolist()
    return [c for c in cols if c != ID_COLUMN]


def summarise(df: pd.DataFrame) -> None:
    print("=== shape")
    print(f"  {len(df)} customers, {df.shape[1]} columns")

    print("\n=== variable types")
    for col in df.columns:
        kind = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"
        if col == ID_COLUMN:
            kind = "identifier"
        if col == TARGET:
            kind = "target"
        print(f"  {col:<26} {kind}")

    print("\n=== missing values")
    missing = df.isna().sum()
    if missing.sum() == 0:
        print("  none")
    else:
        print(missing[missing > 0].to_string())

    print("\n=== numeric summary")
    print(df[numeric_columns(df)].describe().round(2).to_string())

    print("\n=== target balance")
    counts = df[TARGET].value_counts()
    for label, n in counts.items():
        print(f"  {label:<14} {n:>3}  ({n / len(df):.0%})")
    print(f"  majority-class baseline for the model: {counts.max() / len(df):.0%}")


def plot_target_balance(df: pd.DataFrame, out: Path) -> None:
    counts = df[TARGET].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values, color="#4c72b0")
    ax.set_title("Product category distribution (the prediction target)")
    ax.set_xlabel("Product category")
    ax.set_ylabel("Number of customers")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.2, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_distributions(df: pd.DataFrame, out: Path) -> None:
    cols = numeric_columns(df)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, col in zip(axes.flat, cols):
        ax.hist(df[col], bins=15, color="#55a868", edgecolor="white")
        ax.set_title(col)
    for ax in axes.flat[len(cols):]:
        ax.axis("off")
    fig.suptitle("Distribution of each numeric feature")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_outliers(df: pd.DataFrame, out: Path) -> None:
    cols = numeric_columns(df)
    fig, ax = plt.subplots(figsize=(11, 5))
    # Scales differ a lot (spend in hundreds, ratings 1 to 5), so standardise for one shared axis.
    standardised = (df[cols] - df[cols].mean()) / df[cols].std()
    ax.boxplot(standardised.values, labels=cols, vert=True)
    ax.set_title("Outliers per numeric feature (standardised to a shared scale)")
    ax.set_ylabel("Standard deviations from the mean")
    ax.axhline(0, color="grey", linewidth=0.8)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_correlations(df: pd.DataFrame, out: Path) -> None:
    cols = numeric_columns(df)
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Correlation between numeric features")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _outlier_count(series: pd.Series) -> int:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())


def interpret(df: pd.DataFrame) -> None:
    cols = numeric_columns(df)
    corr = df[cols].corr()

    # Strongest off-diagonal pair.
    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(cols) for b in cols[i + 1:]]
    a, b, r = max(pairs, key=lambda p: abs(p[2]))

    skewed = [c for c in cols if df[c].skew() > 0.5]
    outliers = {c: _outlier_count(df[c]) for c in cols if _outlier_count(df[c]) > 0}

    counts = df[TARGET].value_counts()

    print("\n=== reading the plots (draft interpretation for the report)")
    print(f"  Target balance: {counts.idxmax()} is the most common category at "
          f"{counts.max() / len(df):.0%}, so any model has to beat that baseline to be useful.")
    print(f"  Distributions: right-skewed features are {', '.join(skewed)}, a few customers sit "
          f"well above the rest.")
    print(f"  Outliers: the IQR rule flags {', '.join(f'{c} ({n})' for c, n in outliers.items())}. "
          f"These are real high-activity customers, not errors, so they stay in.")
    print(f"  Correlation: the strongest link is {a} and {b} at r={r:.2f}. Otherwise the features "
          f"are weakly correlated, which fits the modest accuracy the product model reaches.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EDA on the merged dataset")
    parser.add_argument("--merged", type=Path, default=config.MERGED_CSV)
    parser.add_argument("--out-dir", type=Path, default=config.ROOT / "reports")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.merged)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summarise(df)

    plot_target_balance(df, args.out_dir / "eda_target_balance.png")
    plot_distributions(df, args.out_dir / "eda_distributions.png")
    plot_outliers(df, args.out_dir / "eda_outliers.png")
    plot_correlations(df, args.out_dir / "eda_correlations.png")

    interpret(df)

    print(f"\nsaved 4 plots to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
