"""
Feature engineering for the return-risk scorer.

Two things live here:
  1. `engineer_features()` - pure, leakage-free, row-wise feature construction
     (delivery mismatch, missingness flags, interaction terms). Safe to apply
     to the whole dataset before splitting because it never looks at the label.
  2. `TargetMeanEncoder` - a fit/transform categorical encoder that must be
     fit ONLY on the training fold and reused via a Pipeline/CV, because it
     DOES use the label. Fitting it on the full dataset before splitting
     would leak test-set label information into the encoding - the single
     most common tabular-ML mistake this project is designed to avoid.

`load_and_split()` is the ONE place the final train/test holdout is created.
Every downstream script (train.py, evaluate.py, explain.py) imports the split
from here so the held-out test set is only ever touched once, at final
evaluation time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

DATA_PATH = "data/orders.csv"
LABEL = "is_returned"
TEST_SIZE = 0.20
SPLIT_SEED = 2024

FIT_SENSITIVE_CATEGORIES = {"fashion", "footwear"}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- delivery mismatch + missingness handling ---
    df["delivery_data_missing"] = df["actual_delivery_days"].isna().astype(int)
    df["delivery_gap_days"] = df["actual_delivery_days"] - df["delivery_promise_days"]

    # --- account-tenure derived flag used by several interaction terms ---
    df["is_new_account"] = ((df["is_first_order"] == 1) | (df["account_age_days"] < 30)).astype(int)

    # --- explicit interaction terms (hypothesized in EDA, baked into the generator) ---
    high_value_threshold = df["order_value"].quantile(0.75)
    df["high_order_value"] = (df["order_value"] > high_value_threshold).astype(int)
    df["new_account_x_high_value"] = df["is_new_account"] * df["high_order_value"]

    df["is_fit_sensitive_category"] = df["item_category"].isin(FIT_SENSITIVE_CATEGORIES).astype(int)
    df["high_discount"] = (df["discount_pct"] > 40).astype(int)
    df["fit_category_x_high_discount"] = df["is_fit_sensitive_category"] * df["high_discount"]

    df["is_cod"] = (df["payment_method"] == "cod").astype(int)
    df["late_delivery"] = (df["delivery_gap_days"] > 1).fillna(0).astype(int)
    df["cod_x_late_delivery"] = df["is_cod"] * df["late_delivery"]

    return df


class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    """Smoothed target-mean encoder. Fit on training labels only; safe for CV
    when wrapped in a sklearn Pipeline (refit per fold) - never fit on data
    that includes the held-out test set or a CV validation fold.
    """

    def __init__(self, columns: list[str], smoothing: float = 20.0):
        self.columns = columns
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y: pd.Series):
        y = pd.Series(np.asarray(y), index=X.index)
        self.global_mean_ = y.mean()
        self.mappings_ = {}
        for col in self.columns:
            stats = y.groupby(X[col]).agg(["mean", "count"])
            smoothed = (stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing) / (
                stats["count"] + self.smoothing
            )
            self.mappings_[col] = smoothed.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            mapping = self.mappings_[col]
            X[f"{col}_te"] = X[col].map(mapping).fillna(self.global_mean_)
        return X.drop(columns=self.columns)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_te" for c in self.columns])


# Columns target-encoded inside the CV/training pipeline (label-dependent).
TARGET_ENCODE_COLS = ["item_category", "payment_method", "device_type", "time_of_day"]

# Row-wise numeric/binary features available after engineer_features(), before encoding.
BASE_NUMERIC_FEATURES = [
    "account_age_days", "prior_orders_count", "prior_return_rate",
    "order_value", "item_count", "discount_pct",
    "delivery_promise_days", "actual_delivery_days", "delivery_gap_days",
    "delivery_data_missing", "is_first_order", "is_new_account",
    "high_order_value", "new_account_x_high_value",
    "is_fit_sensitive_category", "high_discount", "fit_category_x_high_discount",
    "is_cod", "late_delivery", "cod_x_late_delivery",
]

ALL_MODEL_INPUT_COLS = BASE_NUMERIC_FEATURES + TARGET_ENCODE_COLS


def load_and_split():
    """Single source of truth for the final train/test holdout. Returns raw
    (pre-target-encoding) feature frames + labels; target encoding is fit
    inside train.py's CV pipeline, per fold, to avoid leakage.
    """
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    X = df[ALL_MODEL_INPUT_COLS]
    y = df[LABEL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SPLIT_SEED
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split()
    print(f"Train: {X_train.shape}, positive rate {y_train.mean():.3%}")
    print(f"Test:  {X_test.shape}, positive rate {y_test.mean():.3%}")
    print(f"Features ({len(ALL_MODEL_INPUT_COLS)}): {ALL_MODEL_INPUT_COLS}")
