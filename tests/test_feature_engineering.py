"""Unit tests for feature engineering: engineered columns, dtypes, and the
TargetMeanEncoder fit/transform contract."""
import numpy as np
import pandas as pd
import pytest

from feature_engineering import (
    ALL_MODEL_INPUT_COLS,
    TARGET_ENCODE_COLS,
    TargetMeanEncoder,
    engineer_features,
    load_and_split,
)


@pytest.fixture(scope="module")
def raw_df():
    return pd.read_csv("data/orders.csv")


def test_engineer_features_adds_expected_columns(raw_df):
    out = engineer_features(raw_df)
    expected_new_cols = {
        "delivery_data_missing", "delivery_gap_days", "is_new_account",
        "high_order_value", "new_account_x_high_value", "is_fit_sensitive_category",
        "high_discount", "fit_category_x_high_discount", "is_cod", "late_delivery",
        "cod_x_late_delivery",
    }
    assert expected_new_cols.issubset(set(out.columns))


def test_engineered_binary_flags_are_zero_or_one(raw_df):
    out = engineer_features(raw_df)
    binary_cols = [
        "delivery_data_missing", "is_new_account", "high_order_value",
        "new_account_x_high_value", "is_fit_sensitive_category", "high_discount",
        "fit_category_x_high_discount", "is_cod", "late_delivery", "cod_x_late_delivery",
    ]
    for col in binary_cols:
        assert set(out[col].unique()).issubset({0, 1}), f"{col} has non-binary values"


def test_interaction_terms_never_exceed_their_components(raw_df):
    out = engineer_features(raw_df)
    assert (out["new_account_x_high_value"] <= out["is_new_account"]).all()
    assert (out["new_account_x_high_value"] <= out["high_order_value"]).all()
    assert (out["fit_category_x_high_discount"] <= out["is_fit_sensitive_category"]).all()
    assert (out["cod_x_late_delivery"] <= out["is_cod"]).all()


def test_delivery_gap_is_nan_exactly_when_missing_flag_is_set(raw_df):
    out = engineer_features(raw_df)
    missing_mask = out["delivery_data_missing"] == 1
    assert out.loc[missing_mask, "delivery_gap_days"].isna().all()
    assert out.loc[~missing_mask, "delivery_gap_days"].notna().all()


def test_load_and_split_is_stratified_and_reproducible():
    X_train1, X_test1, y_train1, y_test1 = load_and_split()
    X_train2, X_test2, y_train2, y_test2 = load_and_split()
    pd.testing.assert_frame_equal(X_train1, X_train2)
    assert y_train1.mean() == pytest.approx(y_test1.mean(), abs=0.03)


class TestTargetMeanEncoder:
    def test_fit_transform_produces_te_columns_and_drops_originals(self):
        X = pd.DataFrame({"cat": ["a", "b", "a", "b", "c"], "other": [1, 2, 3, 4, 5]})
        y = pd.Series([1, 0, 1, 0, 0])
        enc = TargetMeanEncoder(columns=["cat"], smoothing=1.0)
        enc.fit(X, y)
        out = enc.transform(X)
        assert "cat" not in out.columns
        assert "cat_te" in out.columns
        assert "other" in out.columns

    def test_unseen_category_at_transform_falls_back_to_global_mean(self):
        X_train = pd.DataFrame({"cat": ["a", "a", "b", "b"]})
        y_train = pd.Series([1, 1, 0, 0])
        enc = TargetMeanEncoder(columns=["cat"], smoothing=1.0)
        enc.fit(X_train, y_train)

        X_new = pd.DataFrame({"cat": ["never_seen_before"]})
        out = enc.transform(X_new)
        assert out["cat_te"].iloc[0] == pytest.approx(enc.global_mean_)

    def test_smoothing_pulls_rare_category_toward_global_mean(self):
        # category 'rare' has only 1 observation with label=1 (looks like 100%
        # return rate); heavy smoothing should pull it well below 1.0
        X = pd.DataFrame({"cat": ["common"] * 20 + ["rare"]})
        y = pd.Series([0] * 10 + [1] * 10 + [1])
        enc = TargetMeanEncoder(columns=["cat"], smoothing=50.0)
        enc.fit(X, y)
        assert enc.mappings_["cat"]["rare"] < 0.9
