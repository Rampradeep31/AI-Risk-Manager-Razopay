# Robustness Checks

Produced by [src/robustness.py](src/robustness.py); raw numbers in
[reports/robustness_results.json](reports/robustness_results.json).

## 1. Leakage audit

Every feature that reaches the model was checked against one question:
*"would this be known at order-placement time, or does it require information
from after the fact?"* The full per-feature reasoning is in
[FEATURE_DICTIONARY.md](FEATURE_DICTIONARY.md); the summary table:

| Feature | Known at order time? |
|---|---|
| `account_age_days`, `prior_orders_count`, `prior_return_rate`, `is_first_order` | Yes - customer record as of *before* this order |
| `order_value`, `item_count`, `discount_pct`, `item_category`, `payment_method`, `device_type`, `time_of_day` | Yes - this order's own checkout contents |
| `delivery_promise_days` | Yes - quoted at checkout |
| `actual_delivery_days` | Conditionally - only populated once delivery completes; explicitly modeled as missing (`delivery_data_missing` flag) otherwise, never backfilled from a later date |

No feature derived from the outcome itself (e.g. an actual return flag or
return-processing timestamp) is in the model. The empirical check backs this
up: **max |correlation| between any model-input feature and the label is
0.17** - nowhere near the >0.9 threshold that would indicate a leaked
near-duplicate of the label. (Full correlation table in the EDA step,
[reports/eda/correlation_heatmap.png](reports/eda/correlation_heatmap.png).)

## 2. Stress test (handcrafted edge cases)

Four synthetic orders were constructed to check the model reasons over
*combinations* of features rather than flagging on any single one. **All 4
passed:**

| Case | Expected | Predicted proba | Flagged? | Result |
|---|---|---|---|---|
| New account + high value, otherwise clean: electronics/no discount/on-time/card | flagged | 0.618 | Yes | PASS |
| Genuinely high-risk: new account + high value + footwear + 55% discount + late COD | flagged | 0.983 | Yes | PASS |
| Established customer (40 prior orders, 2% return rate) placing a high-value order | not flagged | 0.036 | No | PASS |
| New account, but low value / grocery / fast delivery | not flagged | 0.277 | No | PASS |

The key result is rows 3 and 4: a high-value order from an established
customer, or a new account placing a low-value order, does **not** push the
score over the threshold on its own - it takes the actual interaction (new
account **and** high value together) to flag an order, confirming
`new_account_x_high_value` is earning its place rather than the model
latching onto one marginal feature. Row 1 shows that interaction is
genuinely strong: it crosses the threshold **by itself**, even with every
other signal clean - consistent with it being the model's 3rd-ranked global
SHAP feature (see [reports/shap/global_importance.png](reports/shap/global_importance.png)).
That's an honest reflection of the synthetic generator's design (new-account
+ high-value risk is deliberately category-independent - see
[MODEL_CARD.md](MODEL_CARD.md#correctness-fix-high_order_value-threshold-found-while-preparing-a-live-demo)),
and a defensible one for an *advisory review-queue* signal rather than an
auto-block.

**Note:** row 1's expectation changed from an earlier version of this check.
A bug in `engineer_features()` (documented in MODEL_CARD.md) computed the
high-value threshold as a per-call quantile, which happens to always
evaluate to "not high value" for a single-row input - so every stress case
and every live API call was silently underestimating this interaction until
the fix. Post-fix, row 1 correctly flags; the row's expected outcome was
updated to match, not loosened to make the test pass.

## 3. Seed stability check

The final (Logistic Regression, `class_weight='balanced'`, `C=0.302`)
configuration was refit from scratch on 3 different train/test splits (seeds
2024, 7, 123 - different from any seed used during model selection) to
confirm the reported metrics aren't an artifact of one particular holdout:

| Seed | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| 2024 | 0.6899 | 0.4370 | 0.4114 |
| 7 | 0.7081 | 0.4398 | 0.4249 |
| 123 | 0.6957 | 0.4510 | 0.4135 |
| **mean +/- std** | **0.698 +/- 0.008** | **0.443 +/- 0.006** | **0.417 +/- 0.006** |

Variance across seeds is small relative to the metric values themselves
(under 2% relative std on both AUCs) - the model's performance is stable and
not a one-split fluke.
