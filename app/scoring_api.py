"""
FastAPI scoring service for the return-risk model.

POST /score takes order-time features and returns:
  - a risk score (calibration caveat: rank-ordered, not a literal probability
    - see MODEL_CARD.md)
  - the threshold decision (flagged for review or not)
  - a SHAP-grounded explanation: the top contributing features (with their
    actual log-odds contribution) and a natural-language sentence built from
    those numbers via a deterministic template.

The natural-language sentence is intentionally template-based, not an LLM
call: an LLM could optionally be swapped in later purely to rephrase this
same template more fluently, but the underlying claim always comes from
SHAP's numbers, never from a model's guess. Run locally:

    uvicorn app.scoring_api:app --reload --port 8000
"""
import json
import sys
from collections import deque
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from explainer import RiskExplainer  # noqa: E402
from feature_engineering import ALL_MODEL_INPUT_COLS, engineer_features, load_and_split  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "final_model_calibrated.joblib"
EXPLAIN_MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"
THRESHOLD = 0.14  # chosen in calibrate.py via cost-based optimization on calibrated train OOF predictions

app = FastAPI(
    title="Return-Risk Scorer",
    description="Advisory return-risk scoring for orders at checkout time. "
                 "See MODEL_CARD.md for intended use, limitations, and calibration caveats.",
    version="1.0.0",
)

# --- Mount static file directories ---
app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")
app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")), name="static")

# --- In-memory scoring history for dashboard ---
_scoring_history = deque(maxlen=50)
_scoring_stats = {"total_scored": 0, "total_flagged": 0, "sum_risk": 0.0}

_pipeline = None       # calibrated model: used for the risk_score users see
_explain_pipeline = None  # uncalibrated pipeline: used only for SHAP attribution
_explainer = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def _get_explainer():
    # Isotonic calibration only remaps the final score onto a true-probability
    # scale - it doesn't change the logistic regression's coefficients or
    # which features drove the decision. So SHAP explains the simpler
    # uncalibrated pipeline directly rather than the 5-model calibrated
    # ensemble; the "why" is identical either way, but far cheaper to compute.
    global _explainer, _explain_pipeline
    if _explainer is None:
        _explain_pipeline = joblib.load(EXPLAIN_MODEL_PATH)
        X_train, _, _, _ = load_and_split()
        _explainer = RiskExplainer(_explain_pipeline, background_X=X_train)
    return _explainer


class OrderFeatures(BaseModel):
    account_age_days: int = Field(..., ge=0, description="Days since account creation (0 for first order)")
    prior_orders_count: int = Field(..., ge=0)
    prior_return_rate: float = Field(..., ge=0.0, le=1.0)
    order_value: float = Field(..., gt=0)
    item_category: Literal["fashion", "footwear", "electronics", "home", "beauty", "grocery", "toys", "sports"]
    item_count: int = Field(..., ge=1)
    discount_pct: float = Field(..., ge=0, le=100)
    payment_method: Literal["card", "upi", "netbanking", "wallet", "cod"]
    delivery_promise_days: int = Field(..., ge=1)
    actual_delivery_days: Optional[int] = Field(None, ge=1, description="Null if order still in transit")
    device_type: Literal["mobile", "desktop", "tablet"]
    time_of_day: Literal["morning", "afternoon", "evening", "late_night"]
    is_first_order: int = Field(..., ge=0, le=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "account_age_days": 0, "prior_orders_count": 0, "prior_return_rate": 0.0,
                "order_value": 4500.0, "item_category": "footwear", "item_count": 1,
                "discount_pct": 55.0, "payment_method": "cod", "delivery_promise_days": 3,
                "actual_delivery_days": 7, "device_type": "mobile", "time_of_day": "evening",
                "is_first_order": 1,
            }
        }
    }


class Contributor(BaseModel):
    feature: str
    friendly_name: str
    shap_value: float
    direction: Literal["increases_risk", "decreases_risk"]


class ScoreResponse(BaseModel):
    risk_score: float
    flagged_for_review: bool
    threshold: float
    top_contributors: list[Contributor]
    explanation: str
    calibration_note: str = (
        "risk_score is isotonic-calibrated on training data (see calibrate.py) - "
        "e.g. a 0.30 score corresponds to roughly a 30% observed return rate in "
        "held-out testing. Still monitor for drift in production; see MODEL_CARD.md."
    )


# --- Root: serve the dashboard ---
@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    html_path = Path(__file__).resolve().parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(order: OrderFeatures):
    raw_df = pd.DataFrame([order.model_dump()])
    engineered = engineer_features(raw_df)[ALL_MODEL_INPUT_COLS]

    pipeline = _get_pipeline()
    proba = float(pipeline.predict_proba(engineered)[:, 1][0])
    flagged = proba >= THRESHOLD

    explainer = _get_explainer()
    exp = explainer.explain(engineered)
    contributors = RiskExplainer.top_contributors(exp[0], top_n=3)
    sentence = RiskExplainer.format_sentence(contributors, proba, flagged)

    # Track in scoring history
    _scoring_stats["total_scored"] += 1
    if flagged:
        _scoring_stats["total_flagged"] += 1
    _scoring_stats["sum_risk"] += proba
    _scoring_history.append({
        "risk_score": round(proba, 4),
        "flagged": flagged,
        "category": order.item_category,
        "payment": order.payment_method,
        "order_value": order.order_value,
    })

    return ScoreResponse(
        risk_score=round(proba, 4),
        flagged_for_review=flagged,
        threshold=THRESHOLD,
        top_contributors=contributors,
        explanation=sentence,
    )


# --- Dashboard data endpoints ---

@app.get("/api/metrics")
def get_metrics():
    """Return model performance metrics from reports."""
    metrics_path = REPORTS_DIR / "final_calibrated" / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    return {"error": "metrics not found"}


@app.get("/api/robustness")
def get_robustness():
    """Return robustness test results."""
    robustness_path = REPORTS_DIR / "robustness_results.json"
    if robustness_path.exists():
        return json.loads(robustness_path.read_text())
    return {"error": "robustness results not found"}


@app.get("/api/sensitivity")
def get_sensitivity():
    """Return threshold sensitivity analysis data."""
    csv_path = REPORTS_DIR / "final_calibrated" / "threshold_sensitivity.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df.to_dict(orient="records")
    return {"error": "sensitivity data not found"}


@app.get("/api/model-info")
def get_model_info():
    """Return model selection and training summary."""
    summary_path = REPORTS_DIR / "model_selection_summary.json"
    comparison_path = REPORTS_DIR / "model_comparison.csv"

    result = {}
    if summary_path.exists():
        result["summary"] = json.loads(summary_path.read_text())
    if comparison_path.exists():
        df = pd.read_csv(comparison_path)
        result["comparison"] = df.to_dict(orient="records")

    return result


@app.get("/api/scoring-history")
def get_scoring_history():
    """Return recent scoring history and stats."""
    return {
        "history": list(_scoring_history),
        "stats": {
            "total_scored": _scoring_stats["total_scored"],
            "total_flagged": _scoring_stats["total_flagged"],
            "flag_rate": round(_scoring_stats["total_flagged"] / max(1, _scoring_stats["total_scored"]), 3),
            "avg_risk": round(_scoring_stats["sum_risk"] / max(1, _scoring_stats["total_scored"]), 4),
        }
    }
