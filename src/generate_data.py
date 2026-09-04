"""
Synthetic data generator for the return-risk scorer.

Design intent (read this before touching the correlations below):
- Fixed seed -> fully reproducible dataset.
- Realistic class imbalance: overall return rate ~18-20%, not 50/50.
- Features are drawn with real-world structure, not independent noise:
    * new accounts and high historical return rate raise risk
    * fashion/footwear/apparel categories raise risk (fit/size issues)
    * large discounts raise risk (impulse buys, "just try it" purchases)
    * late delivery (actual - promised days) raises risk
    * COD / high-friction-refund payment methods interact with category risk
  These are baked in as additive log-odds contributions so a model that only
  learns single-feature thresholds will underperform one that captures the
  INTERACTIONS below (e.g. high order value is only dangerous for new accounts).
- ~6% label noise is flipped in at the end: real return labels are never
  perfectly separable from order-time features, and a model claiming ~99%+
  accuracy here should be treated as a leakage bug, not a triumph.
- ~4% missingness is injected into delivery fields (a realistic gap: actual
  delivery data is sometimes missing at scoring time for in-transit orders).

Calibration reference points used to pick base rates (not joined to the data,
just used to keep the synthetic distribution plausible):
  - Fashion/apparel/footwear e-commerce return rates commonly cited in the
    15-30% range; electronics/grocery much lower (mid-single digits to ~10%).
  - New-account / first-order customers show measurably higher return rates
    than established repeat customers in public retail-analytics writeups.
  - IEEE-CIS Fraud Detection and similar tabular risk datasets motivate the
    general shape used here: heavy right-skewed monetary features, sparse
    high-cardinality categoricals, and behavioral aggregates as top features.
"""
import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 9000

CATEGORIES = ["fashion", "footwear", "electronics", "home", "beauty", "grocery", "toys", "sports"]
CATEGORY_BASE_RISK = {  # additive log-odds nudge per category
    "fashion": 0.85,
    "footwear": 0.95,
    "beauty": 0.35,
    "sports": 0.25,
    "toys": 0.05,
    "home": -0.10,
    "electronics": -0.55,
    "grocery": -1.10,
}
PAYMENT_METHODS = ["card", "upi", "netbanking", "wallet", "cod"]
PAYMENT_RISK = {"cod": 0.45, "wallet": 0.05, "upi": -0.05, "card": -0.10, "netbanking": -0.15}
DEVICE_TYPES = ["mobile", "desktop", "tablet"]
DEVICE_RISK = {"mobile": 0.15, "tablet": 0.05, "desktop": -0.10}


def generate(n_rows: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    is_first_order = rng.binomial(1, 0.22, n_rows)

    # account_age_days: first-order customers are age 0; others log-normal-ish tenure
    account_age_days = np.where(
        is_first_order == 1,
        0,
        np.clip(rng.lognormal(mean=5.2, sigma=1.1, size=n_rows), 1, 3000).astype(int),
    )

    # prior_orders_count: 0 for first order, else a small-count distribution
    prior_orders_count = np.where(
        is_first_order == 1,
        0,
        rng.poisson(lam=np.clip(account_age_days / 120, 0.3, 25), size=n_rows),
    )

    # prior_return_rate: undefined (0) history for new accounts, else Beta-distributed
    # skewed toward low rates but with a meaningful tail of chronic returners
    prior_return_rate = np.where(
        prior_orders_count == 0,
        0.0,
        rng.beta(a=2.0, b=6.0, size=n_rows),
    )

    item_category = rng.choice(CATEGORIES, size=n_rows, p=[0.16, 0.11, 0.16, 0.14, 0.12, 0.13, 0.09, 0.09])

    # order_value: right-skewed monetary distribution, category-adjusted
    base_value = rng.lognormal(mean=6.6, sigma=0.7, size=n_rows)  # ~ few hundred to few thousand INR
    category_value_mult = pd.Series(item_category).map(
        {"electronics": 2.3, "sports": 1.3, "home": 1.2, "fashion": 0.9,
         "footwear": 1.0, "beauty": 0.6, "toys": 0.7, "grocery": 0.4}
    ).to_numpy()
    order_value = np.round(base_value * category_value_mult, 2)

    item_count = rng.poisson(lam=1.8, size=n_rows) + 1
    discount_pct = np.clip(rng.beta(a=1.5, b=5.0, size=n_rows) * 100, 0, 90)

    payment_method = rng.choice(PAYMENT_METHODS, size=n_rows, p=[0.42, 0.28, 0.10, 0.10, 0.10])

    delivery_promise_days = rng.integers(1, 8, size=n_rows)
    # actual delivery has its own noise plus a systematic late-delivery tail
    delivery_delay_noise = rng.normal(0, 1.1, size=n_rows)
    late_shock = rng.binomial(1, 0.12, size=n_rows) * rng.integers(2, 6, size=n_rows)
    actual_delivery_days = np.clip(
        delivery_promise_days + delivery_delay_noise + late_shock, 1, None
    ).round().astype(int)

    device_type = rng.choice(DEVICE_TYPES, size=n_rows, p=[0.58, 0.32, 0.10])
    time_of_day = rng.choice(["morning", "afternoon", "evening", "late_night"], size=n_rows,
                              p=[0.20, 0.30, 0.35, 0.15])

    # ---- assemble log-odds for the true return probability ----
    logit = np.full(n_rows, -2.75)  # base rate anchor, tuned below to land ~18-20% positive

    # customer history signal (strongest real-world driver)
    logit += 0.9 * is_first_order
    logit += -0.55 * np.log1p(prior_orders_count) * (1 - is_first_order)  # tenure/experience reduces risk
    logit += 2.6 * prior_return_rate  # chronic returners are the single strongest signal

    # category signal
    logit += pd.Series(item_category).map(CATEGORY_BASE_RISK).to_numpy()

    # discount / impulse-buy signal
    logit += 0.018 * discount_pct

    # delivery mismatch signal (late delivery drives returns/refusals)
    delivery_gap = actual_delivery_days - delivery_promise_days
    logit += 0.10 * np.clip(delivery_gap, 0, None)

    # payment + device
    logit += pd.Series(payment_method).map(PAYMENT_RISK).to_numpy()
    logit += pd.Series(device_type).map(DEVICE_RISK).to_numpy()

    # item_count: multi-item orders slightly more likely to have a partial return
    logit += 0.06 * np.clip(item_count - 1, 0, None)

    # ---- interaction effects (the part that separates "learns" from "memorizes") ----
    is_new_account = (is_first_order == 1) | (account_age_days < 30)
    high_value = order_value > np.quantile(order_value, 0.75)
    # high order value is only especially risky when paired with a new account
    logit += 1.1 * (is_new_account & high_value)

    fit_sensitive_category = pd.Series(item_category).isin(["fashion", "footwear"]).to_numpy()
    high_discount = discount_pct > 40
    # large discounts on fit-sensitive categories = classic impulse-buy-then-return pattern
    logit += 0.9 * (fit_sensitive_category & high_discount)

    cod_payment = (payment_method == "cod")
    # COD + late delivery: order refusal at the door compounds with delay
    logit += 0.7 * (cod_payment & (delivery_gap > 1))

    prob = 1 / (1 + np.exp(-logit))
    is_returned = rng.binomial(1, prob)

    # ---- label noise: flip ~6% of labels to simulate real-world label imperfection ----
    flip_mask = rng.random(n_rows) < 0.06
    is_returned = np.where(flip_mask, 1 - is_returned, is_returned)

    df = pd.DataFrame({
        "order_id": [f"ORD{100000 + i}" for i in range(n_rows)],
        "account_age_days": account_age_days,
        "prior_orders_count": prior_orders_count,
        "prior_return_rate": np.round(prior_return_rate, 4),
        "order_value": order_value,
        "item_category": item_category,
        "item_count": item_count,
        "discount_pct": np.round(discount_pct, 2),
        "payment_method": payment_method,
        "delivery_promise_days": delivery_promise_days,
        "actual_delivery_days": actual_delivery_days,
        "device_type": device_type,
        "time_of_day": time_of_day,
        "is_first_order": is_first_order,
        "is_returned": is_returned,
    })

    # ---- realistic missingness: ~4% of actual_delivery_days missing (still in-transit
    # at scoring/export time), correlated slightly with longer promised windows ----
    missing_prob = np.clip(0.02 + 0.01 * (delivery_promise_days > 5), 0, 1)
    missing_mask = rng.random(n_rows) < missing_prob
    df.loc[missing_mask, "actual_delivery_days"] = np.nan

    return df


if __name__ == "__main__":
    df = generate()
    out_path = "data/orders.csv"
    df.to_csv(out_path, index=False)
    pos_rate = df["is_returned"].mean()
    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"Positive (returned) rate: {pos_rate:.3%}")
    print(f"Missing actual_delivery_days: {df['actual_delivery_days'].isna().mean():.3%}")
