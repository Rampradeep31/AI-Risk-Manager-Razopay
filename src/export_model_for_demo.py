"""
Exports every numeric parameter needed to reproduce this project's REAL final
model (not an approximation) in client-side JavaScript, for a publicly
shareable interactive demo. No Python backend can be exposed publicly from
this sandboxed environment, so the demo runs the model's actual math -
logistic regression coefficients, target-encoding maps, scaler stats, and the
isotonic calibration curves - directly in the browser.

Two parameter sets are exported:
  1. `explainer`: the single UNCALIBRATED pipeline's parameters, used to
     reproduce the exact SHAP LinearExplainer contribution formula:
         contribution_i = coef_i * (x_i_scaled - background_mean_i)
     (verified bit-for-bit against shap's own output before writing this).
  2. `ensemble`: the 5 CalibratedClassifierCV sub-models (each a full
     pipeline refit on a training fold) + each one's isotonic calibrator.
     The served risk_score is the mean of the 5 folds' calibrated
     probabilities - exactly what models/final_model_calibrated.joblib
     computes, just reimplemented in JS instead of called over a network.

Output: reports/demo_model_export.json
"""
import json

import joblib
import numpy as np

from feature_engineering import (
    BASE_NUMERIC_FEATURES,
    TARGET_ENCODE_COLS,
    FIT_SENSITIVE_CATEGORIES,
    FEATURE_CONSTANTS_PATH,
    load_and_split,
)

FEATURE_NAMES = BASE_NUMERIC_FEATURES + [f"{c}_te" for c in TARGET_ENCODE_COLS]

CATEGORY_OPTIONS = ["fashion", "footwear", "electronics", "home", "beauty", "grocery", "toys", "sports"]
PAYMENT_OPTIONS = ["card", "upi", "netbanking", "wallet", "cod"]
DEVICE_OPTIONS = ["mobile", "desktop", "tablet"]
TIME_OF_DAY_OPTIONS = ["morning", "afternoon", "evening", "late_night"]

FRIENDLY_NAMES = {
    "account_age_days": "account age",
    "prior_orders_count": "number of prior orders",
    "prior_return_rate": "customer's historical return rate",
    "order_value": "order value",
    "item_count": "number of items in order",
    "discount_pct": "discount applied",
    "delivery_promise_days": "promised delivery window",
    "actual_delivery_days": "actual delivery time",
    "delivery_gap_days": "delivery lateness (actual vs. promised)",
    "delivery_data_missing": "delivery data missing (order in transit)",
    "is_first_order": "first-ever order",
    "is_new_account": "new customer account",
    "high_order_value": "high order value",
    "new_account_x_high_value": "new account placing a high-value order",
    "is_fit_sensitive_category": "fit-sensitive category (fashion/footwear)",
    "high_discount": "large discount (>40%)",
    "fit_category_x_high_discount": "large discount on a fit-sensitive item",
    "is_cod": "cash-on-delivery payment",
    "late_delivery": "late delivery",
    "cod_x_late_delivery": "cash-on-delivery order that arrived late",
    "item_category_te": "item category's historical return rate",
    "payment_method_te": "payment method's historical return rate",
    "device_type_te": "device type's historical return rate",
    "time_of_day_te": "time-of-day's historical return rate",
}


def export_pipeline_params(pipe):
    target_enc = pipe.named_steps["target_enc"]
    impute = pipe.named_steps["impute"]
    scale = pipe.named_steps["scale"]
    clf = pipe.named_steps["clf"]
    return {
        "target_enc_mappings": {col: {str(k): v for k, v in target_enc.mappings_[col].items()}
                                 for col in TARGET_ENCODE_COLS},
        "target_enc_global_mean": target_enc.global_mean_,
        "impute_statistics": impute.statistics_.tolist(),
        "scale_mean": scale.mean_.tolist(),
        "scale_scale": scale.scale_.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
    }


def main():
    X_train, X_test, y_train, y_test = load_and_split()

    with open(FEATURE_CONSTANTS_PATH) as f:
        high_value_threshold = json.load(f)["high_value_threshold"]

    # --- 1. explainer params (single uncalibrated pipeline) ---
    uncalibrated = joblib.load("models/final_model.joblib")
    explainer_params = export_pipeline_params(uncalibrated)
    X_train_proc = uncalibrated[:-1].transform(X_train)
    explainer_params["background_mean"] = X_train_proc.mean(axis=0).tolist()

    # --- 2. ensemble params (5-fold calibrated) ---
    calibrated = joblib.load("models/final_model_calibrated.joblib")
    ensemble = []
    for cc in calibrated.calibrated_classifiers_:
        sub_pipe = cc.estimator
        iso = cc.calibrators[0]
        entry = export_pipeline_params(sub_pipe)
        entry["isotonic"] = {
            "x_thresholds": iso.X_thresholds_.tolist(),
            "y_thresholds": iso.y_thresholds_.tolist(),
            "x_min": float(iso.X_min_),
            "x_max": float(iso.X_max_),
        }
        ensemble.append(entry)

    export = {
        "feature_names": FEATURE_NAMES,
        "base_numeric_features": BASE_NUMERIC_FEATURES,
        "target_encode_cols": TARGET_ENCODE_COLS,
        "fit_sensitive_categories": sorted(FIT_SENSITIVE_CATEGORIES),
        "high_value_threshold": high_value_threshold,
        "decision_threshold": 0.14,
        "category_options": CATEGORY_OPTIONS,
        "payment_options": PAYMENT_OPTIONS,
        "device_options": DEVICE_OPTIONS,
        "time_of_day_options": TIME_OF_DAY_OPTIONS,
        "friendly_names": FRIENDLY_NAMES,
        "explainer": explainer_params,
        "ensemble": ensemble,
    }

    out_path = "reports/demo_model_export.json"
    with open(out_path, "w") as f:
        json.dump(export, f)
    print(f"Wrote {out_path}")

    # --- sanity check: reproduce predict_proba for 5 test rows using ONLY the exported params ---
    def isotonic_predict(x, iso_params):
        xs = np.array(iso_params["x_thresholds"])
        ys = np.array(iso_params["y_thresholds"])
        x_clipped = np.clip(x, iso_params["x_min"], iso_params["x_max"])
        return float(np.interp(x_clipped, xs, ys))

    def compute_logit(row_dict, params):
        base_vec = [row_dict[f] for f in BASE_NUMERIC_FEATURES]
        te_vec = []
        for col in TARGET_ENCODE_COLS:
            mapping = params["target_enc_mappings"][col]
            raw_val = row_dict[col]
            te_vec.append(mapping.get(str(raw_val), params["target_enc_global_mean"]))
        full_vec = np.array(base_vec + te_vec, dtype=float)
        stats = np.array(params["impute_statistics"])
        nan_mask = np.isnan(full_vec)
        full_vec[nan_mask] = stats[nan_mask]
        scaled = (full_vec - np.array(params["scale_mean"])) / np.array(params["scale_scale"])
        logit = np.dot(np.array(params["coef"]), scaled) + params["intercept"]
        return logit, scaled

    check_rows = X_test.reset_index(drop=True).iloc[:5]
    true_calibrated_proba = calibrated.predict_proba(X_test.iloc[:5])[:, 1]
    print("\n=== Sanity check: exported-params reproduction vs. real model ===")
    for i in range(5):
        row = check_rows.iloc[i].to_dict()
        fold_probas = []
        for params in ensemble:
            logit, _ = compute_logit(row, params)
            cal_p = isotonic_predict(logit, params["isotonic"])
            fold_probas.append(cal_p)
        reproduced = float(np.mean(fold_probas))
        print(f"  row {i}: reproduced={reproduced:.6f}  actual={true_calibrated_proba[i]:.6f}  "
              f"diff={abs(reproduced - true_calibrated_proba[i]):.2e}")


if __name__ == "__main__":
    main()
