"""
Leakage guard: fails the build if any model-input feature becomes suspiciously
correlated with the label (e.g. someone accidentally adds a post-hoc outcome
field later). Mirrors the manual check in src/robustness.py, but as an
assertion so it runs in CI rather than relying on a human reading printed
output.
"""
import numpy as np

from feature_engineering import ALL_MODEL_INPUT_COLS, LABEL, load_and_split

LEAKAGE_CORR_THRESHOLD = 0.9


def test_label_not_in_feature_columns():
    assert LABEL not in ALL_MODEL_INPUT_COLS


def test_no_feature_exceeds_leakage_correlation_threshold():
    X_train, X_test, y_train, y_test = load_and_split()

    corr_df = X_train.copy()
    for cat_col in ["item_category", "payment_method", "device_type", "time_of_day"]:
        corr_df[cat_col] = corr_df[cat_col].astype("category").cat.codes
    corr_df[LABEL] = y_train.values

    numeric_cols = [c for c in corr_df.columns if c != LABEL]
    label_corr = corr_df[numeric_cols + [LABEL]].corr()[LABEL].drop(LABEL)

    offenders = label_corr[label_corr.abs() > LEAKAGE_CORR_THRESHOLD]
    assert len(offenders) == 0, (
        f"Features with suspiciously high |corr| with label (possible leakage): "
        f"{offenders.to_dict()}"
    )


def test_actual_delivery_days_missingness_is_flagged_not_imputed_silently():
    """actual_delivery_days may be missing (order in transit); the model must
    see an explicit flag rather than have missingness silently coerced away
    before it reaches the pipeline."""
    X_train, X_test, y_train, y_test = load_and_split()
    combined = X_train
    missing_mask = combined["actual_delivery_days"].isna()
    if missing_mask.any():
        assert combined.loc[missing_mask, "delivery_data_missing"].eq(1).all()
    assert combined.loc[~missing_mask, "delivery_data_missing"].eq(0).all()
