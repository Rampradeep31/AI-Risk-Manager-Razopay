"""
Exploratory data analysis for the return-risk dataset.
Run after generate_data.py. Saves plots to reports/eda/ and prints an audit
summary (class balance, missingness, correlation/leakage check) to stdout.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

DATA_PATH = "data/orders.csv"
OUT_DIR = "reports/eda"


def main():
    df = pd.read_csv(DATA_PATH)
    label = "is_returned"

    # ---- 1. class balance ----
    balance = df[label].value_counts(normalize=True).sort_index()
    print("=== Class balance ===")
    print(balance.to_string())
    print(f"Positive rate: {balance.get(1, 0.0):.3%}\n")

    fig, ax = plt.subplots(figsize=(5, 4))
    df[label].value_counts().sort_index().plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_xticklabels(["Not returned (0)", "Returned (1)"], rotation=0)
    ax.set_ylabel("Order count")
    ax.set_title(f"Class balance (positive rate = {balance.get(1, 0.0):.1%})")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/class_balance.png", dpi=150)
    plt.close(fig)

    # ---- 2. missing value audit ----
    missing = df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    print("=== Missing value audit ===")
    if len(missing):
        print(missing.to_string())
    else:
        print("No missing values.")
    print()

    # ---- 3. feature distributions by label ----
    numeric_features = [
        "account_age_days", "prior_orders_count", "prior_return_rate",
        "order_value", "discount_pct", "delivery_promise_days",
        "actual_delivery_days", "item_count",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, feat in zip(axes.ravel(), numeric_features):
        for cls, color in zip([0, 1], ["#4C72B0", "#DD8452"]):
            subset = df.loc[df[label] == cls, feat].dropna()
            ax.hist(subset, bins=30, alpha=0.5, density=True, label=f"label={cls}", color=color)
        ax.set_title(feat)
        ax.legend(fontsize=7)
    fig.suptitle("Feature distributions by label (density-normalized)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/feature_distributions_by_label.png", dpi=150)
    plt.close(fig)

    # categorical: return rate by category
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, feat in zip(axes, ["item_category", "payment_method", "device_type"]):
        rate = df.groupby(feat)[label].mean().sort_values(ascending=False)
        rate.plot(kind="bar", ax=ax, color="#55A868")
        ax.set_ylabel("Return rate")
        ax.set_title(f"Return rate by {feat}")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/return_rate_by_category.png", dpi=150)
    plt.close(fig)

    # ---- 4. correlation matrix / leakage check ----
    corr_df = df.copy()
    for cat_col in ["item_category", "payment_method", "device_type", "time_of_day"]:
        corr_df[cat_col] = corr_df[cat_col].astype("category").cat.codes
    corr_cols = numeric_features + ["is_first_order", "item_category", "payment_method",
                                     "device_type", "time_of_day", label]
    corr = corr_df[corr_cols].corr()

    print("=== Correlation with label (leakage check: nothing should be near +/-1.0) ===")
    label_corr = corr[label].drop(label).sort_values(key=np.abs, ascending=False)
    print(label_corr.to_string())
    suspicious = label_corr[label_corr.abs() > 0.9]
    if len(suspicious):
        print(f"\nWARNING: possible leakage - features with |corr| > 0.9: {list(suspicious.index)}")
    else:
        print("\nNo feature exceeds |corr| > 0.9 with the label - no obvious leakage.")
    print()

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax, square=True,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation matrix (label-encoded categoricals)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/correlation_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"Saved 4 plots to {OUT_DIR}/")


if __name__ == "__main__":
    main()
