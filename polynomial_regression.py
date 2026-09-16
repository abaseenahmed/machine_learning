import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


np.random.seed(42)

# The underlying relationship is quadratic: y = 2x^2 + 3x + 5.
# Small noise makes the train/test evaluation more realistic.
X = np.linspace(-5, 5, 80).reshape(-1, 1)
y = 2 * X[:, 0] ** 2 + 3 * X[:, 0] + 5 + np.random.normal(0, 3, 80)

X_train, X_test, y_train, y_test = train_test_split(
	X,
	y,
	test_size=0.2,
	random_state=42
)


def evaluate_model(degree):
	"""Train a polynomial model and return its predictions and metrics."""
	model = make_pipeline(
		PolynomialFeatures(degree=degree),
		LinearRegression()
	)

	model.fit(X_train, y_train)

	train_prediction = model.predict(X_train)
	test_prediction = model.predict(X_test)

	train_mse = mean_squared_error(y_train, train_prediction)
	test_mse = mean_squared_error(y_test, test_prediction)
	test_r2 = r2_score(y_test, test_prediction)

	print(
		f"Degree {degree}: "
		f"train MSE={train_mse:.2f}, "
		f"test MSE={test_mse:.2f}, "
		f"test R^2={test_r2:.2f}"
	)

	return model


# Comparing degrees shows underfitting, a useful model, and possible overfitting.
models = {degree: evaluate_model(degree) for degree in (1, 2, 5)}

model = models[2]

# PolynomialFeatures adds [1, x, x^2] for degree=2.
feature_names = model.named_steps["polynomialfeatures"].get_feature_names_out(["x"])
coefficients = model.named_steps["linearregression"].coef_
intercept = model.named_steps["linearregression"].intercept_

print("\nGenerated features:", feature_names)
print(f"Learned intercept: {intercept:.2f}")
print("Learned coefficients:", np.round(coefficients, 2))

new_value = np.array([[6]])
print(f"Prediction for x=6: {model.predict(new_value)[0]:.2f}")


# Plot the selected degree-2 model against unseen test data.
X_plot = np.linspace(-5, 6, 300).reshape(-1, 1)
plt.scatter(X_train, y_train, label="Training data")
plt.scatter(X_test, y_test, label="Test data")
plt.plot(X_plot, model.predict(X_plot), color="red", label="Degree-2 model")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Polynomial Regression")
plt.legend()
plt.show()