import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([3, 7, 13, 21, 31])


# Create polynomial features
poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)

print(X_poly)

model = LinearRegression()

model.fit(X_poly, y)

y_predict = model.predict(X_poly)
print(y_predict)