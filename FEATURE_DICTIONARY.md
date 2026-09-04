# Feature Dictionary

All features are known at order-time (no post-hoc / outcome information). See
[ROBUSTNESS.md](ROBUSTNESS.md) for the explicit leakage audit.

## Raw input features

| Feature | Meaning | Why included |
|---|---|---|
| `account_age_days` | Days since account creation (0 for first order) | Tenure is a standard fraud/return risk signal; newer accounts return more |
| `prior_orders_count` | Number of prior completed orders | Distinguishes "new" from "established" customers beyond just tenure |
| `prior_return_rate` | Customer's historical return rate (0 if no prior orders) | Strongest behavioral signal - chronic returners keep returning |
| `order_value` | Order total (currency units) | High-value orders carry more absolute return cost and correlate with impulse buys |
| `item_category` | Product category (fashion, footwear, electronics, ...) | Fit-sensitive categories (fashion/footwear) have structurally higher return rates |
| `item_count` | Number of items in the order | Multi-item orders have more chances for a partial return |
| `discount_pct` | Discount applied, percent | Large discounts correlate with impulse purchases that get returned |
| `payment_method` | card / upi / netbanking / wallet / cod | COD orders can be refused at the door, inflating effective "returns" |
| `delivery_promise_days` | Promised delivery window at order time | Baseline for the delivery-mismatch signal below |
| `actual_delivery_days` | Actual delivery duration (may be missing if in-transit at export time) | Late delivery is a known return/refusal driver |
| `device_type` | mobile / desktop / tablet | Mobile checkout correlates with less-considered purchases |
| `time_of_day` | morning / afternoon / evening / late_night | Late-night orders skew more impulsive |
| `is_first_order` | 1 if this is the customer's first order | Direct new-customer flag; new customers return more |

## Engineered features (`src/feature_engineering.py`)

| Feature | Meaning | Why included |
|---|---|---|
| `delivery_data_missing` | 1 if `actual_delivery_days` is missing (order still in transit at export time) | Explicit missingness indicator so the model (and logistic regression's imputation) doesn't silently treat "missing" as "on time" |
| `delivery_gap_days` | `actual_delivery_days - delivery_promise_days` | Direct measure of how late a delivery was; more informative than the two raw columns separately |
| `is_new_account` | 1 if first order or account age < 30 days | Consolidates two raw signals into one clean "new customer" flag used by the interaction term below |
| `high_order_value` | 1 if `order_value` above the 75th percentile (computed on training data) | Threshold flag used to build the value x tenure interaction |
| `new_account_x_high_value` | `is_new_account * high_order_value` | Interaction: a high-value order is only especially risky when the buyer is new - large orders from established customers are routine |
| `is_fit_sensitive_category` | 1 if category is fashion or footwear | Groups the two categories where size/fit drives returns |
| `high_discount` | 1 if `discount_pct` > 40 | Threshold flag for the discount x category interaction |
| `fit_category_x_high_discount` | `is_fit_sensitive_category * high_discount` | Interaction: heavy discounts on fit-sensitive items are a classic "try it and send it back" pattern; discounts alone (e.g. on electronics) are far less predictive |
| `is_cod` | 1 if payment method is cash-on-delivery | Isolates the payment method with door-refusal risk |
| `late_delivery` | 1 if `delivery_gap_days` > 1 (0 if missing) | Threshold flag for the COD x lateness interaction |
| `cod_x_late_delivery` | `is_cod * late_delivery` | Interaction: COD orders that arrive late are disproportionately refused/returned versus either factor alone |
| `item_category_te`, `payment_method_te`, `device_type_te`, `time_of_day_te` | Smoothed target-mean encoding of each categorical, **fit on the training fold only** (via `TargetMeanEncoder` inside the CV pipeline) | Captures each category's empirical historical return rate as a single ordinal-free numeric feature, avoiding both raw-label-encoding's false ordinality and one-hot's dimensionality blowup; smoothing (m=20) prevents overfitting rare categories |

## Notes on scope

- No `merchant_id` / multi-merchant rollups: this dataset models a single-seller
  storefront, so `prior_return_rate` (customer-level) and the `*_te` columns
  (category/payment/device/time-of-day level) are the only aggregate signals -
  there is no merchant dimension to aggregate over.
- `TargetMeanEncoder` is intentionally NOT fit in `feature_engineering.py`
  itself - it is fit inside each CV fold's pipeline in `train.py` so that no
  validation-fold or test-set label information leaks into the encoding.
