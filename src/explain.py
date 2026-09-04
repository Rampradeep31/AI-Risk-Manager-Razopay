"""
Phase 7 deliverable: SHAP-based explainability for the final model.

Produces:
  reports/shap/global_importance.png   - mean |SHAP value| per feature
  reports/shap/waterfall_case_*.png    - per-prediction breakdown for 2 flagged
                                          test orders and 1 borderline order
  reports/shap/example_explanations.json - structured SHAP output + natural-
                                            language sentence for each example
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from explainer import FEATURE_NAMES, RiskExplainer
from feature_engineering import load_and_split

OUT_DIR = "reports/shap"
THRESHOLD = 0.35  # chosen in evaluate.py


def main():
    X_train, X_test, y_train, y_test = load_and_split()
    pipeline = joblib.load("models/final_model.joblib")

    explainer = RiskExplainer(pipeline, background_X=X_train)

    # ---- global feature importance over the held-out test set ----
    exp_test = explainer.explain(X_test)
    mean_abs_shap = np.abs(exp_test.values).mean(axis=0)
    order = np.argsort(-mean_abs_shap)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh([FEATURE_NAMES[i] for i in order][::-1], mean_abs_shap[order][::-1], color="#4C72B0")
    ax.set_xlabel("Mean |SHAP value| (log-odds contribution, test set)")
    ax.set_title("Global feature importance (SHAP)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/global_importance.png", dpi=150)
    plt.close(fig)

    print("=== Global SHAP importance (top 10) ===")
    for i in order[:10]:
        print(f"  {FEATURE_NAMES[i]:35s} {mean_abs_shap[i]:.4f}")

    # ---- pick worked examples: 2 clearly flagged + 1 borderline ----
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    X_test_reset = X_test.reset_index(drop=True)
    proba_series = test_proba

    flagged_idx = np.argsort(-proba_series)[:2]  # two highest-risk orders
    borderline_idx = [int(np.argmin(np.abs(proba_series - THRESHOLD)))]  # closest to threshold
    example_positions = list(flagged_idx) + borderline_idx

    examples_out = []
    for rank, pos in enumerate(example_positions):
        row_df = X_test_reset.iloc[[pos]]
        exp_row_batch = explainer.explain(row_df)
        exp_row = exp_row_batch[0]
        proba = float(proba_series[pos])
        flagged = bool(proba >= THRESHOLD)
        contributors = RiskExplainer.top_contributors(exp_row, top_n=3)
        sentence = RiskExplainer.format_sentence(contributors, proba, flagged)

        label_kind = "borderline" if rank == len(flagged_idx) else "high_risk"
        print(f"\n--- Example {rank+1} ({label_kind}), predicted proba={proba:.3f}, flagged={flagged} ---")
        print(sentence)

        fig = plt.figure(figsize=(8, 5))
        shap.plots.waterfall(exp_row, show=False, max_display=8)
        fig = plt.gcf()
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/waterfall_case_{rank+1}_{label_kind}.png", dpi=150)
        plt.close(fig)

        examples_out.append({
            "example_kind": label_kind,
            "predicted_proba": proba,
            "flagged": flagged,
            "top_contributors": contributors,
            "explanation_sentence": sentence,
        })

    with open(f"{OUT_DIR}/example_explanations.json", "w") as f:
        json.dump(examples_out, f, indent=2)

    print(f"\nSaved global importance plot, {len(example_positions)} waterfall plots, "
          f"and example_explanations.json to {OUT_DIR}/")


if __name__ == "__main__":
    main()
