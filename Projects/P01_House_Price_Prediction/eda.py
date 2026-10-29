
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_csv("data/cleaned_data.csv")

sns.set_theme(style="whitegrid")

# 1. Basic dataset overview
print("Dataset shape:", df.shape)

print("\nColumn data types:")
print(df.dtypes)

print("\nStatistical summary:")
print(df.describe().round(2))

# 2. Distribution of house prices
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="price", bins=50, kde=True)

plt.title("Distribution of House Prices")
plt.xlabel("House Price ($)")
plt.ylabel("Number of Houses")
plt.tight_layout()
plt.show()

# 3. House prices after logarithmic transformation
plt.figure(figsize=(10, 5))
sns.histplot(x=df["price"].apply(lambda x: __import__("numpy").log1p(x)),
             bins=50, kde=True)

plt.title("Distribution of Log-Transformed House Prices")
plt.xlabel("log(Price + 1)")
plt.ylabel("Number of Houses")
plt.tight_layout()
plt.show()


# 4. Living area vs house price
plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=df,
    x="sqft_living",
    y="price",
    alpha=0.5
)

plt.title("Living Area vs House Price")
plt.xlabel("Living Area (square feet)")
plt.ylabel("House Price ($)")
plt.tight_layout()
plt.show()


# 5. Bedrooms vs house price
plt.figure(figsize=(9, 5))
sns.boxplot(
    data=df,
    x="bedrooms",
    y="price"
)

plt.title("House Prices by Number of Bedrooms")
plt.xlabel("Number of Bedrooms")
plt.ylabel("House Price ($)")
plt.tight_layout()
plt.show()


# 6. Correlation between numerical features
numeric_df = df.select_dtypes(include="number")

correlation = numeric_df.corr()
plt.figure(figsize=(12, 9))
sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# 7. Average price by city: top 10 cities
city_prices = (
    df.groupby("city")["price"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .sort_values()
)

plt.figure(figsize=(10, 6))
city_prices.plot(kind="barh")

plt.title("Top 10 Cities by Average House Price")
plt.xlabel("Average House Price ($)")
plt.ylabel("City")
plt.tight_layout()
plt.show()


# 8. Inspect categorical features
print("\nUnique values in categorical columns:")

for col in ["city", "statezip", "date"]:
    print(f"{col}: {df[col].nunique()} unique values")

print("\nMost common cities:")
print(df["city"].value_counts().head(10))

print("\nMost common statezip values:")
print(df["statezip"].value_counts().head(10))

# 9. Inspect the most expensive properties
print("\nTop 10 most expensive properties:")

columns_to_show = [
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "city",
    "statezip",
    "yr_built",
    "yr_renovated"
]

print(
    df.nlargest(10, "price")[columns_to_show]
    .to_string(index=False)
)

