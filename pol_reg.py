import numpy as np
import matplotlib.pyplot as plt

class PolynomialRegression:
    """
    Polynomial Regression using least squares (Normal Equation).
    
    Fits y = w0 + w1*x + w2*x^2 + ... + wn*x^n
    """
    
    def __init__(self, degree=2, learning_rate=0.01, n_iterations=1000, method='normal'):
        """
        Parameters:
        -----------
        degree : int
            Degree of the polynomial
        learning_rate : float
            Learning rate for gradient descent
        n_iterations : int
            Number of iterations for gradient descent
        method : str
            'normal' for normal equation, 'gd' for gradient descent
        """
        self.degree = degree
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.method = method
        self.coefficients = None
        self.mean_ = None
        self.std_ = None
        self.cost_history = []
    
    def _create_polynomial_features(self, X):
        """Create polynomial features [x, x^2, x^3, ..., x^n]."""
        X = X.flatten()
        X_poly = np.column_stack([X ** d for d in range(1, self.degree + 1)])
        return X_poly
    
    def _normalize(self, X, fit=False):
        """Standardize features for numerical stability."""
        if fit:
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0)
            self.std_[self.std_ == 0] = 1  # avoid division by zero
        return (X - self.mean_) / self.std_
    
    def _add_bias(self, X):
        """Add bias term (column of ones)."""
        return np.column_stack([np.ones(X.shape[0]), X])
    
    def fit(self, X, y):
        """Fit the polynomial regression model."""
        X = np.asarray(X).flatten()
        y = np.asarray(y).flatten()
        
        # Create polynomial features
        X_poly = self._create_polynomial_features(X)
        
        # Normalize features
        X_poly = self._normalize(X_poly, fit=True)
        
        # Add bias term
        X_design = self._add_bias(X_poly)
        
        if self.method == 'normal':
            # Normal equation: w = (X^T X)^(-1) X^T y
            # Using pseudo-inverse for numerical stability
            self.coefficients = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y
        else:
            # Gradient descent
            self.coefficients = self._gradient_descent(X_design, y)
        
        return self
    
    def _gradient_descent(self, X, y):
        """Gradient descent optimization."""
        m, n = X.shape
        w = np.zeros(n)
        self.cost_history = []
        
        for i in range(self.n_iterations):
            y_pred = X @ w
            error = y_pred - y
            gradient = (2 / m) * (X.T @ error)
            w -= self.learning_rate * gradient
            
            # Record cost
            cost = np.mean(error ** 2)
            self.cost_history.append(cost)
        
        return w
    
    def predict(self, X):
        """Predict using the fitted model."""
        X = np.asarray(X).flatten()
        X_poly = self._create_polynomial_features(X)
        X_poly = self._normalize(X_poly, fit=False)
        X_design = self._add_bias(X_poly)
        return X_design @ self.coefficients
    
    def score(self, X, y):
        """R² score."""
        y = np.asarray(y).flatten()
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - (ss_res / ss_tot)
    # Generate synthetic nonlinear data
np.random.seed(42)
X = np.linspace(-3, 3, 100)
y = 0.5 * X**3 - 2 * X**2 + X + 3 + np.random.randn(100) * 2

# Split into train/test
split = 80
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Fit polynomial regression
model = PolynomialRegression(degree=3, method='normal')
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print(f"R² Score: {model.score(X_test, y_test):.4f}")
print(f"Coefficients: {model.coefficients}")

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Training data')
plt.scatter(X_test, y_test, color='green', alpha=0.5, label='Test data')

X_plot = np.linspace(-3, 3, 200)
y_plot = model.predict(X_plot)
plt.plot(X_plot, y_plot, color='red', linewidth=2, label=f'Degree {model.degree} fit')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()