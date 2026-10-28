import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/data.csv")

# 1. Inspect the target variable
print("\nPRICE SUMMARY")
print(df["price"].describe())

print("\nPrices equal to or below zero:")
print((df["price"] <= 0).sum())

print("\nNumber of houses above $2 million:")
print((df["price"] > 2_000_000).sum())

print("\nMost expensive houses:")
print(
    df.nlargest(10, "price")[
        ["price", "bedrooms", "bathrooms", "sqft_living",
         "sqft_lot", "city", "yr_built"]
    ]
)

# 2. Inspect suspicious property values
columns_to_check = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
    "floors", "waterfront", "view", "condition",
    "sqft_above", "sqft_basement", "yr_built", "yr_renovated"
]

print("\nMinimum values:")
print(df[columns_to_check].min())

print("\nMaximum values:")
print(df[columns_to_check].max())

# 3. Check relationships between area columns
print("\nLiving area consistency:")
print(
    (df["sqft_living"] -
     df["sqft_above"] -
     df["sqft_basement"]).describe()
)

# 4. Inspect categorical columns
for col in ["city", "statezip", "country"]:
    print(f"\nUnique values in {col}:")
    print(df[col].value_counts().head(10))

# 5. Convert date to datetime
df["date"] = pd.to_datetime(df["date"], errors="coerce")

print("\nDate range:")
print(df["date"].min(), "to", df["date"].max())

# 6. Visualize the price distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df["price"], bins=50, ax=axes[0])
axes[0].set_title("House Price Distribution")
axes[0].set_xlabel("Price")

sns.boxplot(x=df["price"], ax=axes[1])
axes[1].set_title("House Price Outliers")

plt.tight_layout()
plt.show()

# 7. Visualize price against living area
sns.scatterplot(
    data=df,
    x="sqft_living",
    y="price",
    alpha=0.5
)

plt.title("House Price vs Living Area")
plt.tight_layout()
plt.show()