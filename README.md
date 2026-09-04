# Return-Risk Scorer

An order-time model that predicts the probability an order will be returned,
built with the rigor of a real risk-modeling exercise: honest (imbalanced,
noisy) synthetic data, a fair cross-validated comparison across 3 model
families, an explicit false-positive/false-negative cost model driving
threshold selection, SHAP-grounded per-decision explanations, and documented
robustness checks. See [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md) for the
decision this score informs and who acts on it, and
[MODEL_CARD.md](MODEL_CARD.md) for the full model card.

## Architecture

```
src/generate_data.py        -> data/orders.csv (9,000 rows, seeded, 22.6% return rate)
        |
src/eda.py                  -> reports/eda/*.png (class balance, distributions, correlation/leakage check)
        |
src/feature_engineering.py  -> engineered features + leakage-free train/test split (imported everywhere else)
        |
src/train.py                 -> 5-fold CV: 3 model families -> imbalance handling -> Optuna tuning
        |                        reports/model_comparison.csv, imbalance_comparison.csv,
        |                        optuna_trials.csv, model_selection_summary.json
        |                        models/final_model.joblib (fit on train only)
        |
src/evaluate.py              -> cost-based threshold (train OOF) + final held-out test metrics
        |                        reports/final/{cost_curve,pr_curve,confusion_matrix,calibration_curve}.png, metrics.json
        |
src/explainer.py + explain.py -> SHAP global importance + per-case explanations
        |                        reports/shap/*.png, example_explanations.json
        |
src/robustness.py            -> leakage audit + stress tests + seed-stability check
        |                        reports/robustness_results.json, ROBUSTNESS.md
        |
app/scoring_api.py           -> FastAPI: order features in -> score + threshold decision + SHAP explanation out
```

## Setup

```bash
pip install -r requirements.txt
```

## Running the full pipeline (in order)

```bash
python src/generate_data.py
python src/eda.py
python src/feature_engineering.py
python src/train.py
python src/evaluate.py
python src/explain.py
python src/robustness.py
```

Then serve the scoring API:

```bash
uvicorn app.scoring_api:app --reload --port 8000
```

`POST /score` with an order's features returns a risk score, the
threshold decision, and a SHAP-grounded explanation. Try it via the
auto-generated docs at `http://127.0.0.1:8000/docs`, or:

```bash
curl -X POST http://127.0.0.1:8000/score -H "Content-Type: application/json" -d '{
  "account_age_days": 0, "prior_orders_count": 0, "prior_return_rate": 0.0,
  "order_value": 4500.0, "item_category": "footwear", "item_count": 1,
  "discount_pct": 55.0, "payment_method": "cod", "delivery_promise_days": 3,
  "actual_delivery_days": 7, "device_type": "mobile", "time_of_day": "evening",
  "is_first_order": 1
}'
```

## Key results

- **Model chosen: Logistic Regression** (`class_weight='balanced'`, `C=0.302`),
  selected over Random Forest and LightGBM by 5-fold CV PR-AUC - and it also
  happens to be the most interpretable of the three, so no accuracy/
  interpretability trade-off was needed. Full reasoning (the DGP is
  log-odds-linear with interactions already materialized as columns, which
  removes trees' usual structural advantage) is in
  [reports/model_selection_summary.json](reports/model_selection_summary.json)
  and this repo's phase-by-phase history.
- **Imbalance handling:** `class_weight='balanced'` kept over SMOTE/SMOTE-Tomek
  - the resampling variants didn't clear a 0.005 PR-AUC margin, so the
    simpler, synthetic-data-free option won.
- **Held-out test metrics @ threshold 0.35:** Precision 0.27, Recall 0.85,
  F1 0.41, ROC-AUC 0.69, PR-AUC 0.437 (vs. 0.226 no-skill baseline).
- **Threshold chosen via expected-cost minimization** (FP=Rs50, FN=Rs350) on
  training-set out-of-fold predictions only, then applied once to the
  untouched test set - a 13.4% cost reduction vs. the naive 0.5 threshold.
- **Calibration finding:** the score is reliably rank-ordered but not
  literally calibrated (class-weighting inflates predicted probabilities) -
  documented explicitly rather than glossed over. See
  [MODEL_CARD.md](MODEL_CARD.md#calibration-note-read-before-treating-the-score-as-a-literal-probability).
- **Robustness:** no feature exceeds 0.17 correlation with the label (no
  leakage), 4/4 handcrafted stress tests passed (new-account-alone and
  high-value-alone both correctly fail to trigger a flag; only their
  combination does), and metrics are stable across 3 independent train/test
  splits (ROC-AUC 0.698 +/- 0.008). Full detail in
  [ROBUSTNESS.md](ROBUSTNESS.md).

## Repo map

| Path | What it is |
|---|---|
| [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md) | The decision this score informs, and the cost of getting it wrong |
| [FEATURE_DICTIONARY.md](FEATURE_DICTIONARY.md) | Every feature, what it means, why it's included |
| [MODEL_CARD.md](MODEL_CARD.md) | Intended use, final metrics, threshold reasoning, known failure modes |
| [ROBUSTNESS.md](ROBUSTNESS.md) | Leakage audit, stress tests, seed-stability results |
| `src/` | Data generation, EDA, feature engineering, training, evaluation, explainability, robustness |
| `app/scoring_api.py` | FastAPI scoring service |
| `reports/` | All generated plots and metrics artifacts |
| `models/final_model.joblib` | The fitted, final pipeline |
