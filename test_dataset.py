import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

X = np.linspace(-3, 3, 30)

y = 2 * X**2 + 3 * X + 5 + np.random.normal(0, 3, 30)

plt.scatter(X, y)

plt.xlabel("X")
plt.ylabel("y")
plt.title("Synthetic Dataset")

plt.show()