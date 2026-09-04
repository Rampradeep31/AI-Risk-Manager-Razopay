"""
Final held-out evaluation for the return-risk scorer.

The test set produced by feature_engineering.load_and_split() is touched
EXACTLY ONCE, in this script, and only after the operating threshold has
already been chosen from training-data-only signal:

  1. Cost model: explicit assumed INR costs for a false positive (friction on
     a legitimate order) and a false negative (a missed real return).
  2. Threshold selection: 5-fold out-of-fold (OOF) predicted probabilities on
     the TRAINING set (via cross_val_predict, never touching X_test) are used
     to build an expected-cost-vs-threshold curve; the threshold minimizing
     total expected cost is selected there.
  3. Final evaluation: the already-fitted final_model.joblib (fit on the full
     training set in train.py) scores X_test once; precision/recall/F1,
     confusion matrix, PR curve, ROC-AUC/PR-AUC, and a calibration curve are
     all computed on that single held-out pass.

Doing threshold selection on OOF training predictions instead of the test
set avoids the common mistake of tuning a threshold against the very data
used to report final performance.
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.calibration import calibration_curve
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

from feature_engineering import load_and_split

OUT_DIR = "reports/final"

# --- explicit cost model assumptions (INR per order) ---
# False positive: a legitimate order gets flagged for review. Cost = the
# support/ops handling overhead of the extra check, NOT a lost sale (order
# still ships) - modeled as a flat friction cost.
COST_FALSE_POSITIVE = 50.0
# False negative: an order that will actually be returned ships un-flagged.
# Cost = reverse shipping + restocking/inspection labor + lost margin on an
# order that generated no retained sale.
COST_FALSE_NEGATIVE = 350.0


def select_threshold_via_oof_cost(pipeline, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = cross_val_predict(clone(pipeline), X_train, y_train, cv=cv, method="predict_proba")[:, 1]

    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in thresholds:
        preds = (oof_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_train, preds).ravel()
        total_cost = fp * COST_FALSE_POSITIVE + fn * COST_FALSE_NEGATIVE
        costs.append(total_cost)
    costs = np.array(costs)
    best_idx = costs.argmin()
    best_threshold = thresholds[best_idx]

    # also compute cost at naive 0.5 threshold for comparison
    naive_preds = (oof_proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_train, naive_preds).ravel()
    naive_cost = fp * COST_FALSE_POSITIVE + fn * COST_FALSE_NEGATIVE

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, costs, color="#4C72B0")
    ax.axvline(best_threshold, color="#DD8452", linestyle="--",
               label=f"chosen threshold = {best_threshold:.2f}")
    ax.scatter([best_threshold], [costs[best_idx]], color="#DD8452", zorder=5)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel(f"Expected total cost on train OOF predictions "
                   f"(FP=Rs{COST_FALSE_POSITIVE:.0f}, FN=Rs{COST_FALSE_NEGATIVE:.0f})")
    ax.set_title("Expected cost vs. threshold (5-fold OOF on training data)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/cost_curve.png", dpi=150)
    plt.close(fig)

    return best_threshold, float(costs[best_idx]), float(naive_cost)


def main():
    X_train, X_test, y_train, y_test = load_and_split()
    pipeline = joblib.load("models/final_model.joblib")

    print("=== Step 1: selecting operating threshold from training-set OOF predictions ===")
    best_threshold, best_cost, naive_cost = select_threshold_via_oof_cost(pipeline, X_train, y_train)
    print(f"  Chosen threshold: {best_threshold:.2f}")
    print(f"  Expected cost at chosen threshold (train OOF): Rs{best_cost:,.0f}")
    print(f"  Expected cost at naive 0.5 threshold (train OOF): Rs{naive_cost:,.0f}")
    print(f"  Cost reduction vs naive 0.5: {(1 - best_cost / naive_cost):.1%}\n")

    print("=== Step 2: final evaluation on held-out test set (touched once) ===")
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_preds = (test_proba >= best_threshold).astype(int)

    precision = precision_score(y_test, test_preds)
    recall = recall_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)
    roc_auc = roc_auc_score(y_test, test_proba)
    pr_auc = average_precision_score(y_test, test_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    test_expected_cost = fp * COST_FALSE_POSITIVE + fn * COST_FALSE_NEGATIVE

    print(f"  Precision: {precision:.4f}   Recall: {recall:.4f}   F1: {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}   PR-AUC: {pr_auc:.4f}")
    print(f"  Confusion matrix -> TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"  Test-set expected cost at chosen threshold: Rs{test_expected_cost:,.0f}\n")

    # --- confusion matrix plot ---
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, test_preds),
                            display_labels=["Not returned", "Returned"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix @ threshold={best_threshold:.2f} (test set)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150)
    plt.close(fig)

    # --- precision-recall curve (test set), with operating point marked ---
    prec_curve, rec_curve, pr_thresholds = precision_recall_curve(y_test, test_proba)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(rec_curve, prec_curve, color="#4C72B0", label=f"PR curve (AUC={pr_auc:.3f})")
    ax.scatter([recall], [precision], color="#DD8452", zorder=5,
               label=f"operating point (t={best_threshold:.2f})")
    ax.axhline(y_test.mean(), color="gray", linestyle=":", label=f"no-skill baseline ({y_test.mean():.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (held-out test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/pr_curve.png", dpi=150)
    plt.close(fig)

    # --- calibration curve (test set) ---
    frac_pos, mean_pred = calibration_curve(y_test, test_proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(mean_pred, frac_pos, marker="o", color="#4C72B0", label="model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability (per bin)")
    ax.set_ylabel("Observed return rate (per bin)")
    ax.set_title("Calibration curve (held-out test set, quantile-binned)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/calibration_curve.png", dpi=150)
    plt.close(fig)

    metrics = {
        "chosen_threshold": float(best_threshold),
        "cost_false_positive_inr": COST_FALSE_POSITIVE,
        "cost_false_negative_inr": COST_FALSE_NEGATIVE,
        "train_oof_expected_cost_at_chosen_threshold": best_cost,
        "train_oof_expected_cost_at_naive_0.5": naive_cost,
        "train_oof_cost_reduction_vs_naive": 1 - best_cost / naive_cost,
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "test_roc_auc": float(roc_auc),
        "test_pr_auc": float(pr_auc),
        "test_confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "test_expected_cost_at_chosen_threshold": float(test_expected_cost),
        "test_positive_rate": float(y_test.mean()),
    }
    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved plots + metrics.json to {OUT_DIR}/")


if __name__ == "__main__":
    main()
