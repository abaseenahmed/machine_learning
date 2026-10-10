import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 1. Load
df = pd.read_csv('cleaned_dataset.csv')
print("Data loaded. Shape:", df.shape)

# 2. Clues and answer
X = df.drop(columns=['high_earner', 'salary'])
y = df['high_earner']

# 3. Words → numbers
X = pd.get_dummies(X)

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. ⭐ SCALE the numbers (this fixes the warning!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 6. Train the robot (max_iter bumped to 2000 for safety)
model = LogisticRegression(max_iter=2000, random_state=42)
model.fit(X_train, y_train)

# 7. Test
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))