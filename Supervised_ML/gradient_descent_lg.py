import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PART 1: GRADIENT DESCENT USING SCIKIT-LEARN
# ============================================

def sklearn_gradient_descent_demo():
    """
    Demonstrate Gradient Descent for Linear Regression using scikit-learn
    """
    print("="*60)
    print("GRADIENT DESCENT FOR LINEAR REGRESSION USING SCIKIT-LEARN")
    print("="*60)
    
    # ============================================
    # 1. Generate Sample Data
    # ============================================
    print("\n1. Generating Sample Data...")
    
    # Generate synthetic regression dataset
    X, y, coef = make_regression(
        n_samples=200, 
        n_features=1, 
        noise=20, 
        coef=True, 
        random_state=42
    )
    
    print(f"   - Samples: {X.shape[0]}")
    print(f"   - Features: {X.shape[1]}")
    print(f"   - True coefficient: {coef:.2f}")
    print(f"   - Noise level: 20")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # ============================================
    # 2. Method 1: LinearRegression (Closed-form)
    # ============================================
    print("\n" + "-"*40)
    print("2. Method 1: LinearRegression (Closed-form Solution)")
    print("-"*40)
    
    # Fit LinearRegression (uses normal equations, not gradient descent)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Predictions
    y_pred_lr = lr.predict(X_test)
    
    # Metrics
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    
    # Handle intercept properly (it might be an array)
    intercept_lr = lr.intercept_ if np.isscalar(lr.intercept_) else lr.intercept_[0]
    coef_lr = lr.coef_[0] if len(lr.coef_.shape) > 0 else lr.coef_
    
    print(f"   - Intercept: {intercept_lr:.4f}")
    print(f"   - Coefficient: {coef_lr:.4f}")
    print(f"   - MSE: {mse_lr:.4f}")
    print(f"   - R² Score: {r2_lr:.4f}")
    
    # ============================================
    # 3. Method 2: SGDRegressor (Stochastic GD)
    # ============================================
    print("\n" + "-"*40)
    print("3. Method 2: SGDRegressor (Stochastic Gradient Descent)")
    print("-"*40)
    
    # Scale data for SGD
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SGD Regressor with different learning rates
    sgd_models = {
        'constant': SGDRegressor(
            learning_rate='constant', 
            eta0=0.01,
            max_iter=1000,
            tol=1e-4,
            random_state=42
        ),
        'optimal': SGDRegressor(
            learning_rate='optimal',
            max_iter=1000,
            tol=1e-4,
            random_state=42
        ),
        'invscaling': SGDRegressor(
            learning_rate='invscaling',
            eta0=0.01,
            max_iter=1000,
            tol=1e-4,
            random_state=42
        ),
        'adaptive': SGDRegressor(
            learning_rate='adaptive',
            eta0=0.01,
            max_iter=1000,
            tol=1e-4,
            random_state=42
        )
    }
    
    results = {}
    
    for name, model in sgd_models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Handle intercept properly
        intercept = model.intercept_ if np.isscalar(model.intercept_) else model.intercept_[0]
        coef = model.coef_[0] if len(model.coef_.shape) > 0 else model.coef_
        
        results[name] = {
            'model': model,
            'mse': mse,
            'r2': r2,
            'intercept': intercept,
            'coef': coef,
            'n_iter': model.n_iter_
        }
        
        print(f"\n   {name.capitalize()} Learning Rate:")
        print(f"      - Intercept: {intercept:.4f}")
        print(f"      - Coefficient: {coef:.4f}")
        print(f"      - MSE: {mse:.4f}")
        print(f"      - R² Score: {r2:.4f}")
        print(f"      - Iterations: {model.n_iter_}")
    
    # ============================================
    # 4. Compare Different Loss Functions
    # ============================================
    print("\n" + "-"*40)
    print("4. Comparing Different Loss Functions")
    print("-"*40)
    
    loss_functions = {
        'squared_error': SGDRegressor(
            loss='squared_error',
            learning_rate='optimal',
            max_iter=1000,
            random_state=42
        ),
        'huber': SGDRegressor(
            loss='huber',
            learning_rate='optimal',
            max_iter=1000,
            random_state=42
        ),
        'epsilon_insensitive': SGDRegressor(
            loss='epsilon_insensitive',
            learning_rate='optimal',
            max_iter=1000,
            random_state=42
        )
    }
    
    loss_results = {}
    
    for name, model in loss_functions.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        loss_results[name] = {
            'mse': mse,
            'r2': r2
        }
        
        print(f"\n   {name.replace('_', ' ').title()}:")
        print(f"      - MSE: {mse:.4f}")
        print(f"      - R² Score: {r2:.4f}")
    
    # ============================================
    # 5. Visualization
    # ============================================
    print("\n" + "-"*40)
    print("5. Visualization of Results")
    print("-"*40)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Data and model fits
    axes[0, 0].scatter(X_test, y_test, alpha=0.5, label='Test Data')
    
    # Sort X for smooth line
    X_sorted = np.sort(X_test, axis=0)
    y_pred_lr_sorted = lr.predict(X_sorted)
    y_pred_sgd_sorted = sgd_models['optimal'].predict(scaler.transform(X_sorted))
    
    axes[0, 0].plot(X_sorted, y_pred_lr_sorted, 'r-', linewidth=2, label='LinearRegression')
    axes[0, 0].plot(X_sorted, y_pred_sgd_sorted, 'g--', linewidth=2, label='SGDRegressor')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_title('Model Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. MSE comparison
    models = ['LinearRegression'] + list(sgd_models.keys())
    mse_values = [mse_lr] + [results[name]['mse'] for name in sgd_models.keys()]
    
    axes[0, 1].bar(models, mse_values, color=['blue', 'green', 'orange', 'red', 'purple'])
    axes[0, 1].set_xlabel('Model')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].set_title('MSE Comparison')
    axes[0, 1].tick_params(axis='x', rotation=15)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. R² comparison
    r2_values = [r2_lr] + [results[name]['r2'] for name in sgd_models.keys()]
    
    axes[1, 0].bar(models, r2_values, color=['blue', 'green', 'orange', 'red', 'purple'])
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].set_title('R² Score Comparison')
    axes[1, 0].tick_params(axis='x', rotation=15)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Convergence (for SGD models)
    axes[1, 1].axis('off')
    axes[1, 1].text(0.1, 0.5, 
                   f"Best SGD Model: optimal\n"
                   f"  - Intercept: {results['optimal']['intercept']:.4f}\n"
                   f"  - Coefficient: {results['optimal']['coef']:.4f}\n"
                   f"  - MSE: {results['optimal']['mse']:.4f}\n"
                   f"  - R²: {results['optimal']['r2']:.4f}\n"
                   f"  - Iterations: {results['optimal']['n_iter']}",
                   fontsize=12, verticalalignment='center')
    
    plt.tight_layout()
    plt.show()
    
    return X, y, lr, sgd_models, results


# ============================================
# PART 2: COMPARISON WITH MANUAL IMPLEMENTATION
# ============================================

def manual_gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    """
    Manual implementation of Gradient Descent for comparison
    """
    m = len(y)
    theta = np.random.randn(2, 1) * 0.1
    X_bias = np.c_[np.ones((m, 1)), X]
    history = {'cost': [], 'params': []}
    
    for i in range(iterations):
        # Forward pass
        predictions = X_bias.dot(theta)
        error = predictions - y.reshape(-1, 1)
        
        # Compute gradient
        gradient = (1/m) * X_bias.T.dot(error)
        
        # Update parameters
        theta = theta - learning_rate * gradient
        
        # Store history
        cost = (1/(2*m)) * np.sum(error ** 2)
        history['cost'].append(float(cost))
        history['params'].append(theta.copy())
        
    return theta, history


def compare_methods():
    """
    Compare scikit-learn implementation with manual implementation
    """
    print("\n" + "="*60)
    print("COMPARISON: SCIKIT-LEARN vs MANUAL IMPLEMENTATION")
    print("="*60)
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(100, 1) * 2
    true_theta = np.array([[3.5], [2.0]])
    y = 3.5 + 2 * X + np.random.randn(100, 1) * 0.5
    
    # Manual implementation
    print("\n1. Manual Gradient Descent Implementation:")
    print("-"*40)
    theta_manual, history = manual_gradient_descent(X, y, learning_rate=0.05, iterations=500)
    print(f"   - Intercept (θ₀): {theta_manual[0][0]:.4f}")
    print(f"   - Coefficient (θ₁): {theta_manual[1][0]:.4f}")
    print(f"   - Final cost: {history['cost'][-1]:.4f}")
    
    # Scikit-learn implementation
    print("\n2. Scikit-learn Implementation:")
    print("-"*40)
    lr_sklearn = LinearRegression()
    lr_sklearn.fit(X, y)
    
    intercept_sk = lr_sklearn.intercept_ if np.isscalar(lr_sklearn.intercept_) else lr_sklearn.intercept_[0]
    coef_sk = lr_sklearn.coef_[0] if len(lr_sklearn.coef_.shape) > 0 else lr_sklearn.coef_
    
    print(f"   - Intercept (θ₀): {intercept_sk:.4f}")
    print(f"   - Coefficient (θ₁): {coef_sk:.4f}")
    
    # SGD Regressor
    print("\n3. Scikit-learn SGDRegressor:")
    print("-"*40)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    sgd = SGDRegressor(learning_rate='optimal', max_iter=1000, random_state=42)
    sgd.fit(X_scaled, y.ravel())
    
    # Inverse transform coefficients
    coef_scaled = sgd.coef_[0] / scaler.scale_[0]
    intercept_sgd = sgd.intercept_ - (coef_scaled * scaler.mean_[0])
    
    # Handle intercept properly
    intercept_sgd = intercept_sgd if np.isscalar(intercept_sgd) else intercept_sgd[0]
    
    print(f"   - Intercept (θ₀): {intercept_sgd:.4f}")
    print(f"   - Coefficient (θ₁): {coef_scaled:.4f}")
    print(f"   - Iterations: {sgd.n_iter_}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Cost convergence
    axes[0, 0].plot(history['cost'])
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Cost')
    axes[0, 0].set_title('Cost Convergence (Manual GD)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Parameter trajectory
    params = np.array(history['params'])
    axes[0, 1].plot(params[:, 0, 0], params[:, 1, 0], 'b-', alpha=0.6)
    axes[0, 1].scatter(params[:, 0, 0], params[:, 1, 0], 
                      c=range(len(params)), cmap='viridis', 
                      s=20, alpha=0.7)
    axes[0, 1].scatter(params[0, 0, 0], params[0, 1, 0], 
                      color='red', s=100, marker='o', label='Start')
    axes[0, 1].scatter(params[-1, 0, 0], params[-1, 1, 0], 
                      color='green', s=100, marker='*', label='End')
    axes[0, 1].set_xlabel('θ₀')
    axes[0, 1].set_ylabel('θ₁')
    axes[0, 1].set_title('Parameter Trajectory')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Data and fitted line (Manual)
    axes[1, 0].scatter(X, y, alpha=0.5, label='Data')
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line_manual = theta_manual[0][0] + theta_manual[1][0] * X_line
    axes[1, 0].plot(X_line, y_line_manual, 'r-', linewidth=2, label='Manual GD')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_title('Manual GD Fit')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Data and fitted line (Scikit-learn)
    axes[1, 1].scatter(X, y, alpha=0.5, label='Data')
    y_line_sklearn = lr_sklearn.predict(X_line)
    axes[1, 1].plot(X_line, y_line_sklearn, 'g-', linewidth=2, label='Scikit-learn')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].set_title('Scikit-learn Fit')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================
# PART 3: HYPERPARAMETER TUNING
# ============================================

def tune_sgd_hyperparameters():
    """
    Demonstrate hyperparameter tuning for SGD Regressor
    """
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING FOR SGD REGRESSOR")
    print("="*60)
    
    # Generate data
    X, y = make_regression(n_samples=500, n_features=5, noise=10, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test different hyperparameters
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    penalties = ['l2', 'l1', 'elasticnet']
    alphas = [0.0001, 0.001, 0.01, 0.1]
    
    results = []
    
    for lr in learning_rates:
        for penalty in penalties:
            for alpha in alphas[:2]:  # Limit for demonstration
                sgd = SGDRegressor(
                    learning_rate='constant',
                    eta0=lr,
                    penalty=penalty,
                    alpha=alpha,
                    max_iter=1000,
                    random_state=42
                )
                
                sgd.fit(X_train_scaled, y_train)
                y_pred = sgd.predict(X_test_scaled)
                
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results.append({
                    'learning_rate': lr,
                    'penalty': penalty,
                    'alpha': alpha,
                    'mse': mse,
                    'r2': r2
                })
    
    # Display best results
    print("\nTop 5 Best Performances:")
    print("-"*50)
    
    sorted_results = sorted(results, key=lambda x: x['mse'])
    for i, result in enumerate(sorted_results[:5]):
        print(f"\n{i+1}. LR={result['learning_rate']}, Penalty={result['penalty']}, Alpha={result['alpha']}")
        print(f"   MSE: {result['mse']:.4f}, R²: {result['r2']:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # MSE by learning rate
    lr_groups = {}
    for r in results:
        lr = r['learning_rate']
        if lr not in lr_groups:
            lr_groups[lr] = []
        lr_groups[lr].append(r['mse'])
    
    axes[0].boxplot(list(lr_groups.values()), labels=[str(lr) for lr in lr_groups.keys()])
    axes[0].set_xlabel('Learning Rate')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('Effect of Learning Rate on MSE')
    axes[0].grid(True, alpha=0.3)
    
    # MSE by penalty
    penalty_groups = {}
    for r in results:
        penalty = r['penalty']
        if penalty not in penalty_groups:
            penalty_groups[penalty] = []
        penalty_groups[penalty].append(r['mse'])
    
    axes[1].boxplot(list(penalty_groups.values()), labels=list(penalty_groups.keys()))
    axes[1].set_xlabel('Penalty')
    axes[1].set_ylabel('MSE')
    axes[1].set_title('Effect of Penalty on MSE')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return sorted_results[:5]


# ============================================
# PART 4: ONLINE LEARNING DEMONSTRATION
# ============================================

def online_learning_demo():
    """
    Demonstrate online learning with SGD
    """
    print("\n" + "="*60)
    print("ONLINE LEARNING WITH SGD REGRESSOR")
    print("="*60)
    
    # Generate streaming data
    np.random.seed(42)
    n_samples = 1000
    X_stream = np.random.randn(n_samples, 1) * 2
    true_coef = 2.5
    true_intercept = 1.0
    
    # Initialize SGD with warm_start for online learning
    sgd = SGDRegressor(
        learning_rate='constant',
        eta0=0.01,
        warm_start=True,
        max_iter=1,
        random_state=42
    )
    
    # Simulate online learning
    batch_size = 50
    predictions = []
    coefficients = []
    intercepts = []
    
    for i in range(0, n_samples, batch_size):
        # Get current batch
        X_batch = X_stream[i:i+batch_size]
        
        # Generate target with noise
        y_batch = true_intercept + true_coef * X_batch + np.random.randn(batch_size, 1) * 0.5
        
        # Fit on batch (continues from previous state)
        sgd.partial_fit(X_batch, y_batch.ravel())
        
        # Store coefficients
        coef = sgd.coef_[0] if len(sgd.coef_.shape) > 0 else sgd.coef_
        intercept = sgd.intercept_ if np.isscalar(sgd.intercept_) else sgd.intercept_[0]
        
        coefficients.append(coef)
        intercepts.append(intercept)
        
        # Make prediction for visualization
        X_test = np.linspace(-4, 4, 100).reshape(-1, 1)
        y_pred = sgd.predict(X_test)
        predictions.append((X_test, y_pred))
    
    # Plot learning evolution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Coefficient evolution
    axes[0, 0].plot(range(len(coefficients)), coefficients)
    axes[0, 0].axhline(y=true_coef, color='r', linestyle='--', label='True coefficient')
    axes[0, 0].set_xlabel('Batch Number')
    axes[0, 0].set_ylabel('Coefficient')
    axes[0, 0].set_title('Coefficient Convergence')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Intercept evolution
    axes[0, 1].plot(range(len(intercepts)), intercepts)
    axes[0, 1].axhline(y=true_intercept, color='r', linestyle='--', label='True intercept')
    axes[0, 1].set_xlabel('Batch Number')
    axes[0, 1].set_ylabel('Intercept')
    axes[0, 1].set_title('Intercept Convergence')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Final fit
    axes[1, 0].scatter(X_stream, true_intercept + true_coef * X_stream, 
                       alpha=0.3, s=10, label='Data')
    axes[1, 0].plot(predictions[-1][0], predictions[-1][1], 
                    'r-', linewidth=2, label='Final model')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_title('Final Model Fit')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Learning progress (first vs last)
    axes[1, 1].scatter(X_stream, true_intercept + true_coef * X_stream, 
                       alpha=0.3, s=10, label='Data')
    axes[1, 1].plot(predictions[0][0], predictions[0][1], 
                    'b--', linewidth=2, label='Initial model', alpha=0.5)
    axes[1, 1].plot(predictions[-1][0], predictions[-1][1], 
                    'r-', linewidth=2, label='Final model')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].set_title('Learning Progress: Initial vs Final')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    final_intercept = sgd.intercept_ if np.isscalar(sgd.intercept_) else sgd.intercept_[0]
    final_coef = sgd.coef_[0] if len(sgd.coef_.shape) > 0 else sgd.coef_
    
    print(f"\nFinal Model:")
    print(f"   - Intercept: {final_intercept:.4f} (True: {true_intercept:.4f})")
    print(f"   - Coefficient: {final_coef:.4f} (True: {true_coef:.4f})")


# ============================================
# PART 5: PRACTICAL USAGE GUIDE
# ============================================

def practical_usage_guide():
    """
    Provide a practical guide for using SGD in scikit-learn
    """
    print("\n" + "="*60)
    print("PRACTICAL USAGE GUIDE FOR SCIKIT-LEARN SGD")
    print("="*60)
    
    print("""
    1. Basic Usage:
    --------------
    from sklearn.linear_model import SGDRegressor
    from sklearn.preprocessing import StandardScaler
    
    # Scale your data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create and train model
    sgd = SGDRegressor(max_iter=1000, random_state=42)
    sgd.fit(X_scaled, y)
    
    # Make predictions
    predictions = sgd.predict(X_scaled_test)
    
    2. Important Parameters:
    -----------------------
    • max_iter: Maximum number of passes over the data
    • learning_rate: 'constant', 'optimal', 'invscaling', 'adaptive'
    • eta0: Initial learning rate (for 'constant' and 'invscaling')
    • penalty: 'l2', 'l1', 'elasticnet' (regularization)
    • alpha: Regularization strength
    • loss: 'squared_error', 'huber', 'epsilon_insensitive'
    • warm_start: Reuse previous solution as initialization
    
    3. Tips for Success:
    -------------------
    • Always scale your features (SGD is sensitive to feature scaling)
    • Start with a small learning rate and increase if needed
    • Use 'adaptive' learning rate to avoid overshooting
    • Monitor training loss to ensure convergence
    • Use early stopping to prevent overfitting
    • Consider using partial_fit for online learning
    
    4. Comparison with Other Models:
    ------------------------------
    • LinearRegression: Closed-form, fast for small datasets
    • SGDRegressor: Better for large datasets, online learning
    • Ridge/Lasso: Regularized linear models with closed-form
    • ElasticNet: Combines L1 and L2 regularization
    
    5. Common Pitfalls:
    ------------------
    • Forgetting to scale features
    • Using too high learning rate (diverge)
    • Using too low learning rate (slow convergence)
    • Not handling categorical variables
    • Ignoring convergence warnings
    """)


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Run all demonstrations
    sklearn_gradient_descent_demo()
    compare_methods()
    tune_sgd_hyperparameters()
    online_learning_demo()
    practical_usage_guide()
    
    print("\n" + "="*60)
    print("SUMMARY: KEY TAKEAWAYS")
    print("="*60)
    print("""
    1. Scikit-learn provides two main ways to do linear regression:
       - LinearRegression: Closed-form solution (not gradient descent)
       - SGDRegressor: Stochastic Gradient Descent
    
    2. SGDRegressor Advantages:
       - Handles large datasets efficiently
       - Supports online learning with partial_fit
       - Provides regularization (L1, L2, ElasticNet)
       - Different learning rate schedules
    
    3. Key Parameters to Tune:
       - learning_rate: Choose based on your data
       - eta0: Start with 0.01, adjust as needed
       - max_iter: Usually 1000-5000
       - penalty: Add regularization to prevent overfitting
    
    4. Always Remember:
       - Scale your features before using SGD
       - Check convergence with validation data
       - Use warm_start for continued learning
       - Monitor both training and validation metrics
    
    5. When to Use SGD vs LinearRegression:
       - SGD: Large datasets, online learning, memory constraints
       - LinearRegression: Small datasets, need exact solution
    """)