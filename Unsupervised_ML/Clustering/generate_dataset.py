"""
Generate a realistic synthetic customer dataset for clustering practice.
Fixed: add size=n to every rng.normal() call.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=42)

segments = {
    "Budget Shoppers": {
        "n": 400,
        "age": (45, 10),
        "annual_income_k": (28, 6),
        "spending_score": (20, 8),
        "purchases_per_year": (6, 2),
        "avg_order_value": (35, 10),
        "days_since_last_purchase": (120, 40),
        "loyalty_years": (2.0, 1.0),
        "returns_rate": (0.12, 0.05),
    },
    "Loyal Mid-Tier": {
        "n": 550,
        "age": (38, 8),
        "annual_income_k": (58, 10),
        "spending_score": (55, 10),
        "purchases_per_year": (18, 5),
        "avg_order_value": (80, 20),
        "days_since_last_purchase": (30, 15),
        "loyalty_years": (5.0, 1.5),
        "returns_rate": (0.07, 0.03),
    },
    "Premium VIP": {
        "n": 250,
        "age": (42, 9),
        "annual_income_k": (120, 25),
        "spending_score": (85, 8),
        "purchases_per_year": (35, 8),
        "avg_order_value": (220, 60),
        "days_since_last_purchase": (12, 7),
        "loyalty_years": (8.0, 2.0),
        "returns_rate": (0.05, 0.02),
    },
    "Impulse Buyers": {
        "n": 350,
        "age": (29, 6),
        "annual_income_k": (55, 12),
        "spending_score": (78, 12),
        "purchases_per_year": (28, 6),
        "avg_order_value": (55, 15),
        "days_since_last_purchase": (20, 10),
        "loyalty_years": (2.5, 1.2),
        "returns_rate": (0.20, 0.07),
    },
    "New / Dormant": {
        "n": 450,
        "age": (32, 11),
        "annual_income_k": (48, 15),
        "spending_score": (30, 12),
        "purchases_per_year": (3, 2),
        "avg_order_value": (45, 20),
        "days_since_last_purchase": (220, 60),
        "loyalty_years": (0.7, 0.4),
        "returns_rate": (0.10, 0.05),
    },
}

rows = []
customer_id = 10001

for segment_name, params in segments.items():
    n = params["n"]

    # ✅ KEY FIX: size=n on every rng.normal call
    age          = np.clip(rng.normal(*params["age"],                    size=n), 18, 75).round(0)
    income       = np.clip(rng.normal(*params["annual_income_k"],        size=n), 10, 200).round(1)
    spend_score  = np.clip(rng.normal(*params["spending_score"],         size=n), 1, 100).round(0)
    purchases    = np.clip(rng.normal(*params["purchases_per_year"],     size=n), 0, 60).round(0)
    order_value  = np.clip(rng.normal(*params["avg_order_value"],        size=n), 5, 500).round(2)
    days_last    = np.clip(rng.normal(*params["days_since_last_purchase"], size=n), 0, 365).round(0)
    loyalty      = np.clip(rng.normal(*params["loyalty_years"],          size=n), 0, 20).round(1)
    returns_rate = np.clip(rng.normal(*params["returns_rate"],           size=n), 0.0, 0.5).round(3)

    revenue = (purchases * order_value).round(2)

    for i in range(n):
        rows.append({
            "CustomerID": customer_id,
            "Age": int(age[i]),
            "AnnualIncome_k": float(income[i]),
            "SpendingScore": int(spend_score[i]),
            "PurchasesPerYear": int(purchases[i]),
            "AvgOrderValue": float(order_value[i]),
            "AnnualRevenue": float(revenue[i]),
            "DaysSinceLastPurchase": int(days_last[i]),
            "LoyaltyYears": float(loyalty[i]),
            "ReturnsRate": float(returns_rate[i]),
            "Segment": segment_name,
        })
        customer_id += 1

df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Mild imperfections
for col in ["AnnualIncome_k", "ReturnsRate"]:
    mask = rng.random(len(df)) < 0.005
    df.loc[mask, col] = np.nan

outlier_idx = rng.choice(df.index, size=5, replace=False)
df.loc[outlier_idx, "AnnualRevenue"] *= rng.uniform(8, 15, size=5)

df.to_csv("customer_segmentation_dataset.csv", index=False)

print(f"✅ Dataset saved: customer_segmentation_dataset.csv")
print(f"   Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\n{df.head()}")
print(f"\nSegment counts:\n{df['Segment'].value_counts()}")