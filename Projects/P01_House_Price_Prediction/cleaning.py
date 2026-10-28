
import pandas as pd

# 1. Load the original dataset
df = pd.read_csv("data/data.csv")

print("Original dataset shape:", df.shape)

# 2. Remove houses with invalid sale prices
df = df[df["price"] > 0].copy()

print("After removing zero-price records:", df.shape)

# 3. Remove records with both zero bedrooms and zero bathrooms
invalid_rooms = (
    (df["bedrooms"] == 0) &
    (df["bathrooms"] == 0)
)

print("Suspicious room records found:", invalid_rooms.sum())

df = df.loc[~invalid_rooms].copy()

print("After removing suspicious records:", df.shape)

# 4. Remove columns we won't initially use
df = df.drop(columns=["country", "street"])

# 5. Convert date from text to datetime
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# 6. Save the cleaned dataset
df.to_csv("data/cleaned_data.csv", index=False)

# 7. Validate the result
print("\nFinal cleaned dataset shape:", df.shape)
print("Remaining nonpositive prices:", (df["price"] <= 0).sum())
print("Remaining missing values:", df.isnull().sum().sum())
print("\nCleaned dataset saved to data/cleaned_data.csv")
