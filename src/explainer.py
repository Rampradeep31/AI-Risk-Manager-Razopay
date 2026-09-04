"""
Reusable SHAP explanation layer for the return-risk scorer.

Used both by explain.py (this phase's deliverable: global importance + a few
worked per-case examples) and by scoring_api.py (per-request explanations at
serve time). The final model is a linear (logistic regression) pipeline, so
shap.LinearExplainer gives exact, fast, analytic Shapley values - no sampling
approximation needed.

The `format_sentence()` template turns raw SHAP numbers into a natural-
language sentence. This is deliberately a plain string template, not an LLM
call: the LLM formatter mentioned in the brief is an OPTIONAL later step that
would only rephrase this same template's output more fluently - it is never
the source of the underlying claim. The numbers (and which features drove
the decision) always come from SHAP, not from a model guessing.
"""
from __future__ import annotations

import numpy as np
import shap

from feature_engineering import BASE_NUMERIC_FEATURES, TARGET_ENCODE_COLS

FEATURE_NAMES = BASE_NUMERIC_FEATURES + [f"{c}_te" for c in TARGET_ENCODE_COLS]

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


class RiskExplainer:
    def __init__(self, pipeline, background_X):
        self.pipeline = pipeline
        self.preprocessor = pipeline[:-1]
        self.classifier = pipeline.named_steps["clf"]
        # Display values shown alongside SHAP numbers should be in
        # human-meaningful (pre-StandardScaler) units, not scaled z-scores -
        # SHAP values themselves are still computed in the scaled space the
        # classifier actually sees, since that's what LinearExplainer needs.
        if "scale" in getattr(pipeline, "named_steps", {}):
            self.display_preprocessor = pipeline[:-2]
        else:
            self.display_preprocessor = self.preprocessor
        background_proc = self.preprocessor.transform(background_X)
        self.explainer = shap.LinearExplainer(self.classifier, background_proc)

    def explain(self, X_rows_df):
        """Returns a shap.Explanation batch for the given (raw, pre-encoding) rows,
        with .data overridden to show pre-scaling (human-readable) feature values."""
        X_proc = self.preprocessor.transform(X_rows_df)
        exp = self.explainer(X_proc)
        exp.feature_names = FEATURE_NAMES
        display_values = np.asarray(self.display_preprocessor.transform(X_rows_df))
        exp.data = display_values
        return exp

    @staticmethod
    def top_contributors(exp_row, top_n: int = 3) -> list[dict]:
        vals = np.asarray(exp_row.values)
        order = np.argsort(-np.abs(vals))[:top_n]
        contributors = []
        for i in order:
            fname = FEATURE_NAMES[i]
            contributors.append({
                "feature": fname,
                "friendly_name": FRIENDLY_NAMES.get(fname, fname),
                "shap_value": float(vals[i]),
                "direction": "increases_risk" if vals[i] > 0 else "decreases_risk",
            })
        return contributors

    @staticmethod
    def format_sentence(contributors: list[dict], predicted_proba: float, flagged: bool) -> str:
        parts = [f"{c['friendly_name']} ({c['shap_value']:+.2f})" for c in contributors]
        verdict = "flagged as high return-risk" if flagged else "not flagged (low return-risk)"
        return (f"Order {verdict} (predicted return probability {predicted_proba:.0%}). "
                f"Top drivers: " + "; ".join(parts) + ".")
