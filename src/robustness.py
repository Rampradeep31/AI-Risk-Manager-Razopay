"""
Phase 8 deliverable: robustness checks for the final model.

  1. Leakage audit: an explicit, per-feature "was this known at order time?"
     table, plus a re-check that no feature is suspiciously (|corr| > 0.9)
     correlated with the label.
  2. Stress test: handcrafted edge-case orders (adversarial-looking but
     legitimate, and genuinely high-risk) to confirm the model reasons over
     combinations of features rather than flagging on a single one.
  3. Stability check: refit the final (model, imbalance, hyperparameters)
     configuration on 3 different train/test splits (different seeds) and
     report variance in test-set ROC-AUC / PR-AUC / F1.

Results are printed and also written to reports/robustness_results.json so
ROBUSTNESS.md can quote exact numbers.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from explainer import RiskExplainer
from feature_engineering import ALL_MODEL_INPUT_COLS, engineer_features, load_and_split
from train import build_pipeline

THRESHOLD = 0.35

# ---------------------------------------------------------------------------
# 1. Leakage audit
# ---------------------------------------------------------------------------
# For every feature that reaches the model: is it known at order-placement
# time, or does it require post-hoc / outcome information? Anything in the
# latter category must never appear in ALL_MODEL_INPUT_COLS.
LEAKAGE_AUDIT = {
    "account_age_days": "known at order time (customer record)",
    "prior_orders_count": "known at order time (customer record, orders BEFORE this one)",
    "prior_return_rate": "known at order time (customer record, computed from orders BEFORE this one)",
    "order_value": "known at order time (this order's own contents)",
    "item_count": "known at order time (this order's own contents)",
    "discount_pct": "known at order time (this order's own contents)",
    "delivery_promise_days": "known at order time (quoted at checkout)",
    "actual_delivery_days": "known at SCORING time only if delivery has completed; "
                            "explicitly modeled as missing otherwise via delivery_data_missing "
                            "- never backfilled from a later date",
    "device_type": "known at order time (checkout session)",
    "time_of_day": "known at order time (checkout timestamp)",
    "is_first_order": "known at order time (customer record)",
    "item_category": "known at order time (this order's own contents)",
    "payment_method": "known at order time (checkout selection)",
}


def leakage_audit(X_train, y_train) -> dict:
    print("=== 1. Leakage audit ===")
    for feat, note in LEAKAGE_AUDIT.items():
        print(f"  [OK] {feat}: {note}")

    corr_df = X_train.copy()
    for cat_col in ["item_category", "payment_method"]:
        if cat_col in corr_df.columns:
            corr_df[cat_col] = corr_df[cat_col].astype("category").cat.codes
    corr_df["is_returned"] = y_train.values
    numeric_corr_cols = [c for c in corr_df.columns if corr_df[c].dtype != object and c != "is_returned"]
    label_corr = corr_df[numeric_corr_cols + ["is_returned"]].corr()["is_returned"].drop("is_returned")
    max_abs_corr = label_corr.abs().max()
    suspicious = label_corr[label_corr.abs() > 0.9]
    print(f"\n  Max |corr(feature, label)| among model-input features: {max_abs_corr:.4f}")
    if len(suspicious):
        print(f"  WARNING - possible leakage: {list(suspicious.index)}")
    else:
        print("  No feature exceeds |corr| > 0.9 with the label - consistent with EDA, no leakage found.")

    return {"max_abs_label_correlation": float(max_abs_corr), "suspicious_features": list(suspicious.index)}


# ---------------------------------------------------------------------------
# 2. Stress test
# ---------------------------------------------------------------------------
def make_order(**overrides) -> dict:
    base = {
        "account_age_days": 400, "prior_orders_count": 12, "prior_return_rate": 0.05,
        "order_value": 800.0, "item_category": "electronics", "item_count": 1,
        "discount_pct": 5.0, "payment_method": "card", "delivery_promise_days": 4,
        "actual_delivery_days": 4, "device_type": "desktop", "time_of_day": "afternoon",
        "is_first_order": 0,
    }
    base.update(overrides)
    return base


STRESS_CASES = [
    {
        "name": "adversarial_but_legitimate_high_value_new_account",
        "description": "New account + high order value, BUT clean category (electronics), "
                        "no discount, on-time delivery, card payment - should NOT be flagged "
                        "purely because of the new-account + high-value combination alone.",
        "order": make_order(account_age_days=0, prior_orders_count=0, prior_return_rate=0.0,
                             is_first_order=1, order_value=4500.0, item_category="electronics",
                             discount_pct=0.0, payment_method="card",
                             delivery_promise_days=3, actual_delivery_days=3),
        "expect_flagged": False,
    },
    {
        "name": "genuinely_high_risk",
        "description": "New account + high value + fit-sensitive category + heavy discount + "
                        "late COD delivery - every known risk factor stacked.",
        "order": make_order(account_age_days=0, prior_orders_count=0, prior_return_rate=0.0,
                             is_first_order=1, order_value=4500.0, item_category="footwear",
                             discount_pct=55.0, payment_method="cod",
                             delivery_promise_days=3, actual_delivery_days=7),
        "expect_flagged": True,
    },
    {
        "name": "established_customer_high_value",
        "description": "Established, low-return-history customer placing a high-value order - "
                        "tests that high value alone (without new-account) isn't enough to flag.",
        "order": make_order(account_age_days=900, prior_orders_count=40, prior_return_rate=0.02,
                             is_first_order=0, order_value=4500.0, item_category="electronics",
                             discount_pct=5.0, payment_method="card",
                             delivery_promise_days=3, actual_delivery_days=3),
        "expect_flagged": False,
    },
    {
        "name": "new_account_low_value_clean",
        "description": "New account but low order value, non-fit-sensitive category, fast "
                        "delivery - tests that new-account status alone isn't enough to flag.",
        "order": make_order(account_age_days=0, prior_orders_count=0, prior_return_rate=0.0,
                             is_first_order=1, order_value=250.0, item_category="grocery",
                             discount_pct=0.0, payment_method="upi",
                             delivery_promise_days=2, actual_delivery_days=2),
        "expect_flagged": False,
    },
]


def stress_test(pipeline, explainer) -> list[dict]:
    print("\n=== 2. Stress test (handcrafted edge cases) ===")
    results = []
    for case in STRESS_CASES:
        raw_df = pd.DataFrame([case["order"]])
        engineered = engineer_features(raw_df)[ALL_MODEL_INPUT_COLS]
        proba = float(pipeline.predict_proba(engineered)[:, 1][0])
        flagged = proba >= THRESHOLD
        passed = flagged == case["expect_flagged"]
        exp = explainer.explain(engineered)
        contributors = RiskExplainer.top_contributors(exp[0], top_n=3)

        status = "PASS" if passed else "FAIL"
        print(f"\n  [{status}] {case['name']}")
        print(f"    {case['description']}")
        print(f"    predicted_proba={proba:.3f}  flagged={flagged}  expected_flagged={case['expect_flagged']}")
        print(f"    top drivers: " + ", ".join(f"{c['friendly_name']} ({c['shap_value']:+.2f})"
                                                for c in contributors))
        results.append({
            "name": case["name"], "description": case["description"],
            "predicted_proba": proba, "flagged": flagged,
            "expected_flagged": case["expect_flagged"], "passed": passed,
            "top_contributors": contributors,
        })
    return results


# ---------------------------------------------------------------------------
# 3. Seed stability check
# ---------------------------------------------------------------------------
def stability_check() -> list[dict]:
    print("\n=== 3. Seed stability check (3 different train/test splits) ===")
    with open("reports/model_selection_summary.json") as f:
        config = json.load(f)
    family = config["model_family"]
    imbalance = config["imbalance_strategy"]
    params = config["best_params"]

    results = []
    for seed in [2024, 7, 123]:
        X_train, X_test, y_train, y_test = load_and_split(seed=seed)
        pipe = build_pipeline(family, imbalance=imbalance, params=params)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        preds = (proba >= THRESHOLD).astype(int)
        roc = roc_auc_score(y_test, proba)
        pr = average_precision_score(y_test, proba)
        f1 = f1_score(y_test, preds)
        print(f"  seed={seed:5d}  ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  F1={f1:.4f}")
        results.append({"seed": seed, "roc_auc": roc, "pr_auc": pr, "f1": f1})

    roc_vals = [r["roc_auc"] for r in results]
    pr_vals = [r["pr_auc"] for r in results]
    f1_vals = [r["f1"] for r in results]
    print(f"\n  ROC-AUC: mean={np.mean(roc_vals):.4f} std={np.std(roc_vals):.4f}")
    print(f"  PR-AUC:  mean={np.mean(pr_vals):.4f} std={np.std(pr_vals):.4f}")
    print(f"  F1:      mean={np.mean(f1_vals):.4f} std={np.std(f1_vals):.4f}")
    return results


def main():
    X_train, X_test, y_train, y_test = load_and_split()
    pipeline = joblib.load("models/final_model.joblib")
    explainer = RiskExplainer(pipeline, background_X=X_train)

    leakage_result = leakage_audit(X_train, y_train)
    stress_results = stress_test(pipeline, explainer)
    stability_results = stability_check()

    all_passed = all(r["passed"] for r in stress_results)
    print(f"\n=== Summary: stress tests {'ALL PASSED' if all_passed else 'HAD FAILURES'} ===")

    output = {
        "leakage_audit": leakage_result,
        "stress_test": stress_results,
        "stability_check": stability_results,
    }
    with open("reports/robustness_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved reports/robustness_results.json")


if __name__ == "__main__":
    main()
