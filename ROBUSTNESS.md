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
| Adversarial-but-legitimate: new account + high value, but electronics/no discount/on-time/card | not flagged | 0.327 | No | PASS |
| Genuinely high-risk: new account + high value + footwear + 55% discount + late COD | flagged | 0.947 | Yes | PASS |
| Established customer (40 prior orders, 2% return rate) placing a high-value order | not flagged | 0.034 | No | PASS |
| New account, but low value / grocery / fast delivery | not flagged | 0.276 | No | PASS |

The key result is the first and third rows: a new account alone, or a
high-value order alone, does **not** push the score over the threshold - it
takes the interaction (new account **and** high value, stacked with category/
discount/delivery signals) to flag an order. This confirms the
`new_account_x_high_value`, `fit_category_x_high_discount`, and
`cod_x_late_delivery` interaction features are actually earning their place
rather than the model just latching onto one strong marginal feature.

## 3. Seed stability check

The final (Logistic Regression, `class_weight='balanced'`, `C=0.302`)
configuration was refit from scratch on 3 different train/test splits (seeds
2024, 7, 123 - different from any seed used during model selection) to
confirm the reported metrics aren't an artifact of one particular holdout:

| Seed | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| 2024 | 0.6899 | 0.4370 | 0.4107 |
| 7 | 0.7081 | 0.4394 | 0.4204 |
| 123 | 0.6959 | 0.4511 | 0.4066 |
| **mean +/- std** | **0.698 +/- 0.008** | **0.443 +/- 0.006** | **0.413 +/- 0.006** |

Variance across seeds is small relative to the metric values themselves
(under 2% relative std on both AUCs) - the model's performance is stable and
not a one-split fluke.
