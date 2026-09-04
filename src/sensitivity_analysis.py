"""
Cost-ratio sensitivity analysis for the operating threshold.

evaluate.py / calibrate.py pick ONE threshold from ONE pair of assumed costs
(FP=Rs50, FN=Rs350, ratio 7:1). That pair is a judgment call - a reviewer
will reasonably ask "what if you're wrong about the ratio?" This script
answers that directly: for each candidate cost ratio, on the SAME calibrated
out-of-fold training predictions used elsewhere, it re-derives the
cost-optimal threshold and reports what precision/recall/F1/expected-cost
that threshold would produce on the (still only-touched-once) held-out test
set.

Only the RATIO FN:FP matters for where the argmin falls (scaling both costs
by the same constant leaves the optimal threshold unchanged), so the sweep
varies the ratio directly, holding FP=Rs50 fixed and varying FN.

Outputs:
  reports/final_calibrated/threshold_sensitivity.csv
  reports/final_calibrated/threshold_sensitivity.png
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from evaluate import COST_FALSE_POSITIVE
from feature_engineering import load_and_split

OUT_DIR = "reports/final_calibrated"
BASELINE_RATIO = 350.0 / COST_FALSE_POSITIVE  # 7.0, the ratio used everywhere else

# Sweep from ~0.3x to ~3x the baseline ratio - covers "cost model materially
# wrong in either direction" without wandering into implausible territory.
RATIO_MULTIPLIERS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def best_threshold_for_ratio(oof_proba, y_train, cost_fp, cost_fn):
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = np.array([
        (lambda cm: cm[0, 1] * cost_fp + cm[1, 0] * cost_fn)(
            confusion_matrix(y_train, (oof_proba >= t).astype(int))
        )
        for t in thresholds
    ])
    best_idx = costs.argmin()
    return thresholds[best_idx], float(costs[best_idx])


def main():
    X_train, X_test, y_train, y_test = load_and_split()
    calibrated = joblib.load("models/final_model_calibrated.joblib")

    # Reuse the same calibrated model to score OOF-style: since re-fitting per
    # ratio would be identical (the ratio only affects threshold selection,
    # not model fitting), we generate ONE set of OOF-quality probabilities via
    # the already-fitted calibrated model on X_train. This is a light
    # approximation of true OOF (the model saw X_train during calibration),
    # acceptable here because we're only comparing *relative* threshold
    # movement across cost ratios, not re-deriving a fresh point estimate.
    from sklearn.base import clone
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    import json
    with open("reports/model_selection_summary.json") as f:
        config = json.load(f)
    from train import build_pipeline
    uncalibrated_pipeline = build_pipeline(config["model_family"], config["imbalance_strategy"],
                                            config["best_params"])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = cross_val_predict(
        clone(CalibratedClassifierCV(estimator=uncalibrated_pipeline, method="isotonic", cv=5)),
        X_train, y_train, cv=cv, method="predict_proba",
    )[:, 1]

    test_proba = calibrated.predict_proba(X_test)[:, 1]

    print(f"=== Cost-ratio sensitivity analysis (baseline FN:FP ratio = {BASELINE_RATIO:.1f}) ===")
    rows = []
    for mult in RATIO_MULTIPLIERS:
        ratio = BASELINE_RATIO * mult
        cost_fn = COST_FALSE_POSITIVE * ratio
        threshold, train_oof_cost = best_threshold_for_ratio(oof_proba, y_train, COST_FALSE_POSITIVE, cost_fn)

        test_preds = (test_proba >= threshold).astype(int)
        precision = precision_score(y_test, test_preds, zero_division=0)
        recall = recall_score(y_test, test_preds, zero_division=0)
        f1 = f1_score(y_test, test_preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
        test_cost = fp * COST_FALSE_POSITIVE + fn * cost_fn

        tag = "  <-- baseline" if mult == 1.0 else ""
        print(f"  ratio={ratio:5.1f} (FN=Rs{cost_fn:6.0f})  threshold={threshold:.2f}  "
              f"precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}  "
              f"test_cost=Rs{test_cost:,.0f}{tag}")

        rows.append({
            "ratio_multiplier": mult, "fn_fp_ratio": ratio, "cost_fn_inr": cost_fn,
            "cost_fp_inr": COST_FALSE_POSITIVE, "chosen_threshold": threshold,
            "test_precision": precision, "test_recall": recall, "test_f1": f1,
            "test_expected_cost": test_cost,
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/threshold_sensitivity.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(df["fn_fp_ratio"], df["chosen_threshold"], marker="o", color="#4C72B0")
    ax1.axvline(BASELINE_RATIO, color="#DD8452", linestyle="--", label=f"baseline ratio={BASELINE_RATIO:.1f}")
    ax1.set_xlabel("Assumed FN:FP cost ratio")
    ax1.set_ylabel("Cost-optimal threshold")
    ax1.set_title("How the threshold moves if the cost ratio is wrong")
    ax1.legend()

    ax2.plot(df["fn_fp_ratio"], df["test_precision"], marker="o", label="precision", color="#4C72B0")
    ax2.plot(df["fn_fp_ratio"], df["test_recall"], marker="o", label="recall", color="#55A868")
    ax2.plot(df["fn_fp_ratio"], df["test_f1"], marker="o", label="F1", color="#DD8452")
    ax2.axvline(BASELINE_RATIO, color="gray", linestyle="--")
    ax2.set_xlabel("Assumed FN:FP cost ratio")
    ax2.set_ylabel("Test-set metric")
    ax2.set_title("Resulting precision/recall/F1 (held-out test set)")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/threshold_sensitivity.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved threshold_sensitivity.csv and .png to {OUT_DIR}/")


if __name__ == "__main__":
    main()
