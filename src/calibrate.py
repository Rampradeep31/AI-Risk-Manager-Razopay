"""
Isotonic recalibration of the final model.

evaluate.py found the raw model well rank-ordered but NOT well-calibrated in
absolute terms (class_weight='balanced' inflates predicted probabilities).
This script fixes that:

  1. Wrap the final pipeline in sklearn's CalibratedClassifierCV
     (method='isotonic', cv=5): internally this refits 5 clones of the
     pipeline on 5 training folds and fits an isotonic regressor mapping raw
     score -> calibrated probability on each fold's held-out portion, then
     ensembles the 5 calibrators. No test-set data is involved.
  2. Re-select the operating threshold via the SAME expected-cost
     minimization as evaluate.py, but now on out-of-fold CALIBRATED
     probabilities (threshold selection must use whichever score the
     production system will actually emit).
  3. Evaluate the refit-on-full-training-data calibrated model on the held-out
     test set exactly once, and plot a before/after calibration curve so the
     fix is visible, not just claimed.

SHAP explanations continue to come from the ORIGINAL uncalibrated pipeline
(models/final_model.joblib): isotonic recalibration only remaps the final
aggregate score onto a true-probability scale - it does not change the
logistic regression's coefficients or which features drove a decision, so
feature attribution is unaffected and explaining through the calibrated
ensemble (5 internal sub-models) would only add complexity with no benefit.
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from evaluate import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE
from feature_engineering import load_and_split
from train import build_pipeline

OUT_DIR = "reports/final_calibrated"


def select_threshold_via_oof_cost(estimator, X_train, y_train, seed=42):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_proba = cross_val_predict(clone(estimator), X_train, y_train, cv=cv, method="predict_proba")[:, 1]

    thresholds = np.linspace(0.01, 0.99, 99)
    costs = np.array([
        (lambda cm: cm[0, 1] * COST_FALSE_POSITIVE + cm[1, 0] * COST_FALSE_NEGATIVE)(
            confusion_matrix(y_train, (oof_proba >= t).astype(int))
        )
        for t in thresholds
    ])
    best_idx = costs.argmin()
    return thresholds[best_idx], float(costs[best_idx]), oof_proba


def main():
    X_train, X_test, y_train, y_test = load_and_split()

    with open("reports/model_selection_summary.json") as f:
        config = json.load(f)
    uncalibrated_pipeline = build_pipeline(config["model_family"], config["imbalance_strategy"],
                                            config["best_params"])

    print("=== Step 1: fitting isotonic calibration (5-fold, ensembled) ===")
    calibrated = CalibratedClassifierCV(estimator=uncalibrated_pipeline, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)
    joblib.dump(calibrated, "models/final_model_calibrated.joblib")

    print("=== Step 2: re-selecting threshold on calibrated OOF training predictions ===")
    best_threshold, best_cost, oof_calibrated_proba = select_threshold_via_oof_cost(
        CalibratedClassifierCV(estimator=uncalibrated_pipeline, method="isotonic", cv=5), X_train, y_train
    )
    print(f"  Chosen threshold (calibrated): {best_threshold:.2f}  (was 0.35 pre-calibration)")

    print("\n=== Step 3: final evaluation on held-out test set ===")
    raw_pipeline = joblib.load("models/final_model.joblib")
    raw_test_proba = raw_pipeline.predict_proba(X_test)[:, 1]
    calibrated_test_proba = calibrated.predict_proba(X_test)[:, 1]
    test_preds = (calibrated_test_proba >= best_threshold).astype(int)

    precision = precision_score(y_test, test_preds)
    recall = recall_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)
    roc_auc = roc_auc_score(y_test, calibrated_test_proba)
    pr_auc = average_precision_score(y_test, calibrated_test_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    test_expected_cost = fp * COST_FALSE_POSITIVE + fn * COST_FALSE_NEGATIVE

    print(f"  Precision: {precision:.4f}   Recall: {recall:.4f}   F1: {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}   PR-AUC: {pr_auc:.4f}")
    print(f"  Confusion matrix -> TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"  Test-set expected cost at chosen threshold: Rs{test_expected_cost:,.0f}")

    # --- before/after calibration curve ---
    frac_pos_raw, mean_pred_raw = calibration_curve(y_test, raw_test_proba, n_bins=10, strategy="quantile")
    frac_pos_cal, mean_pred_cal = calibration_curve(y_test, calibrated_test_proba, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(mean_pred_raw, frac_pos_raw, marker="o", color="#DD8452", label="before calibration (raw)")
    ax.plot(mean_pred_cal, frac_pos_cal, marker="o", color="#4C72B0", label="after isotonic calibration")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability (per bin)")
    ax.set_ylabel("Observed return rate (per bin)")
    ax.set_title("Calibration: before vs. after isotonic recalibration (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/calibration_curve_comparison.png", dpi=150)
    plt.close(fig)

    # --- confusion matrix + PR curve at the new threshold ---
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, test_preds),
                            display_labels=["Not returned", "Returned"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix (calibrated) @ threshold={best_threshold:.2f}")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150)
    plt.close(fig)

    prec_curve, rec_curve, _ = precision_recall_curve(y_test, calibrated_test_proba)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(rec_curve, prec_curve, color="#4C72B0", label=f"PR curve (AUC={pr_auc:.3f})")
    ax.scatter([recall], [precision], color="#DD8452", zorder=5, label=f"operating point (t={best_threshold:.2f})")
    ax.axhline(y_test.mean(), color="gray", linestyle=":", label=f"no-skill baseline ({y_test.mean():.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve, calibrated model (held-out test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/pr_curve.png", dpi=150)
    plt.close(fig)

    # --- quantify the calibration improvement with a scalar (mean |gap|) ---
    raw_gap = float(np.mean(np.abs(frac_pos_raw - mean_pred_raw)))
    cal_gap = float(np.mean(np.abs(frac_pos_cal - mean_pred_cal)))

    metrics = {
        "chosen_threshold": float(best_threshold),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "test_roc_auc": float(roc_auc),
        "test_pr_auc": float(pr_auc),
        "test_confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "test_expected_cost_at_chosen_threshold": float(test_expected_cost),
        "mean_abs_calibration_gap_before": raw_gap,
        "mean_abs_calibration_gap_after": cal_gap,
        "calibration_gap_reduction": 1 - cal_gap / raw_gap,
    }
    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Mean |calibration gap| before: {raw_gap:.4f}  after: {cal_gap:.4f}  "
          f"({(1 - cal_gap / raw_gap):.1%} reduction)")
    print(f"Saved calibrated model to models/final_model_calibrated.joblib")
    print(f"Saved plots + metrics.json to {OUT_DIR}/")


if __name__ == "__main__":
    main()
