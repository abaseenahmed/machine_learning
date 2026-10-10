import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PART 1: BASIC GRADIENT DESCENT IMPLEMENTATION
# ============================================

class GradientDescent:
    """
    A comprehensive implementation of Gradient Descent with multiple variants
    """
    
    def __init__(self, learning_rate=0.1, max_iterations=1000, tolerance=1e-6):
        """
        Initialize the Gradient Descent optimizer
        
        Parameters:
        -----------
        learning_rate : float, step size for each iteration
        max_iterations : int, maximum number of iterations
        tolerance : float, stopping criterion (minimum change in cost)
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.history = {'cost': [], 'parameters': []}
        
    def cost_function(self, X, y, theta):
        """
        Mean Squared Error cost function for linear regression
        
        J(θ) = (1/2m) * Σ(hθ(x) - y)²
        """
        m = len(y)
        predictions = X.dot(theta)
        cost = (1/(2*m)) * np.sum((predictions - y) ** 2)
        return cost
    
    def gradient(self, X, y, theta):
        """
        Compute the gradient of the cost function
        
        ∂J/∂θ = (1/m) * Xᵀ(Xθ - y)
        """
        m = len(y)
        predictions = X.dot(theta)
        gradient = (1/m) * X.T.dot(predictions - y)
        return gradient
    
    def fit(self, X, y, verbose=True):
        """
        Run gradient descent to find optimal parameters
        
        Parameters:
        -----------
        X : array, feature matrix (add bias term manually)
        y : array, target values
        verbose : bool, print progress
        """
        # Initialize parameters randomly
        np.random.seed(42)
        theta = np.random.randn(X.shape[1], 1)
        
        # Store initial state
        self.history['parameters'].append(theta.copy())
        self.history['cost'].append(self.cost_function(X, y, theta))
        
        for i in range(self.max_iterations):
            # Compute gradient
            grad = self.gradient(X, y, theta)
            
            # Update parameters
            theta_new = theta - self.learning_rate * grad
            
            # Store history
            self.history['parameters'].append(theta_new.copy())
            self.history['cost'].append(self.cost_function(X, y, theta_new))
            
            # Check convergence
            if np.abs(self.history['cost'][-1] - self.history['cost'][-2]) < self.tolerance:
                if verbose:
                    print(f"Converged after {i+1} iterations")
                break
            
            theta = theta_new
            
            if verbose and i % 100 == 0:
                print(f"Iteration {i}: Cost = {self.history['cost'][-1]:.4f}")
        
        self.theta_optimal = theta
        self.convergence_iterations = len(self.history['cost']) - 1
        
        return theta
    
    def predict(self, X):
        """Make predictions using the trained model"""
        if not hasattr(self, 'theta_optimal'):
            raise ValueError("Model not trained yet. Call fit() first.")
        return X.dot(self.theta_optimal)


# ============================================
# PART 2: GRADIENT DESCENT VARIANTS (FIXED)
# ============================================

def stochastic_gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    """
    Stochastic Gradient Descent - updates parameters for each sample
    
    Returns: trajectory of parameters
    """
    m, n = X.shape
    theta = np.random.randn(n, 1)
    trajectory = [theta.copy()]
    costs = []
    
    for epoch in range(epochs):
        # Shuffle the data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_cost = 0
        for i in range(m):
            # Single sample update
            xi = X_shuffled[i:i+1]
            yi = y_shuffled[i:i+1]
            
            prediction = xi.dot(theta)
            error = prediction - yi
            gradient = xi.T.dot(error)
            
            theta = theta - learning_rate * gradient
            trajectory.append(theta.copy())
            
            # Accumulate squared error for this epoch
            epoch_cost += float(error ** 2)  # Convert to scalar
        
        # Store average cost for this epoch
        costs.append(epoch_cost / m)
    
    return np.array(trajectory), costs  # costs is now a 1D array


def mini_batch_gradient_descent(X, y, learning_rate=0.01, batch_size=32, epochs=1000):
    """
    Mini-Batch Gradient Descent - updates parameters using batches
    
    Returns: trajectory of parameters
    """
    m, n = X.shape
    theta = np.random.randn(n, 1)
    trajectory = [theta.copy()]
    costs = []
    
    for epoch in range(epochs):
        # Shuffle the data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_cost = 0
        num_batches = max(1, m // batch_size)
        
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            predictions = X_batch.dot(theta)
            error = predictions - y_batch
            gradient = X_batch.T.dot(error) / batch_size
            
            theta = theta - learning_rate * gradient
            trajectory.append(theta.copy())
            
            # Accumulate squared error for this batch
            epoch_cost += float(np.sum(error ** 2))  # Convert to scalar
        
        # Store average cost for this epoch
        costs.append(epoch_cost / m)
    
    return np.array(trajectory), costs  # costs is now a 1D array


# ============================================
# PART 3: VISUALIZATION FUNCTIONS
# ============================================

def plot_cost_convergence(history):
    """Plot the cost function convergence over iterations"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Full view
    axes[0].plot(history['cost'])
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Cost (J(θ))')
    axes[0].set_title('Cost Convergence')
    axes[0].grid(True, alpha=0.3)
    
    # Zoomed view of later iterations
    zoom_start = max(0, len(history['cost']) - 50)
    axes[1].plot(range(zoom_start, len(history['cost'])), 
                 history['cost'][zoom_start:])
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Cost (J(θ))')
    axes[1].set_title('Cost Convergence (Zoomed)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_parameter_path(theta_history, true_theta=None):
    """Plot the path of parameters during optimization"""
    theta_history = np.array(theta_history)
    
    if theta_history.shape[1] == 2:  # Only if we have 2 parameters
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot parameter path
        ax.plot(theta_history[:, 0], theta_history[:, 1], 'b-', alpha=0.6, 
                label='Parameter path', linewidth=2)
        ax.scatter(theta_history[:, 0], theta_history[:, 1], 
                  c=range(len(theta_history)), cmap='viridis', 
                  s=30, alpha=0.7, label='Iterations')
        
        # Plot starting and ending points
        ax.scatter(theta_history[0, 0], theta_history[0, 1], 
                  color='red', s=100, marker='o', label='Start', zorder=5)
        ax.scatter(theta_history[-1, 0], theta_history[-1, 1], 
                  color='green', s=100, marker='*', label='End', zorder=5)
        
        if true_theta is not None:
            ax.scatter(true_theta[0], true_theta[1], 
                      color='gold', s=150, marker='X', 
                      label='True optimum', zorder=5)
        
        ax.set_xlabel('θ₀')
        ax.set_ylabel('θ₁')
        ax.set_title('Parameter Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.colorbar(ax.collections[0], label='Iteration')
        plt.show()
    else:
        # For more than 2 parameters, plot each parameter over time
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(min(theta_history.shape[1], 5)):  # Plot max 5 parameters
            ax.plot(theta_history[:, i], label=f'θ_{i}', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Parameter Value')
        ax.set_title('Parameter Evolution Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.show()


# ============================================
# PART 4: DEMONSTRATION WITH EXAMPLE DATA
# ============================================

def create_sample_data(n_samples=100, noise=0.5):
    """Create sample data for linear regression"""
    np.random.seed(42)
    X = 2 * np.random.rand(n_samples, 1)
    true_theta = np.array([[4.0], [3.0]])
    y = 4 + 3 * X + noise * np.random.randn(n_samples, 1)
    
    # Add bias term
    X_bias = np.c_[np.ones((n_samples, 1)), X]
    return X_bias, y, true_theta


def demonstrate_gradient_descent():
    """Complete demonstration of gradient descent"""
    print("="*60)
    print("GRADIENT DESCENT DEMONSTRATION")
    print("="*60)
    
    # Create data
    X, y, true_theta = create_sample_data(n_samples=200, noise=0.3)
    
    print(f"\nDataset created with:")
    print(f"  - Samples: {X.shape[0]}")
    print(f"  - Features: {X.shape[1]-1}")
    print(f"  - True parameters: θ₀={true_theta[0][0]:.2f}, θ₁={true_theta[1][0]:.2f}")
    
    # ---------------------------------------------------------
    # 1. Basic Gradient Descent
    # ---------------------------------------------------------
    print("\n" + "-"*40)
    print("1. Basic Gradient Descent")
    print("-"*40)
    
    gd = GradientDescent(learning_rate=0.05, max_iterations=1000)
    theta_optimal = gd.fit(X, y, verbose=True)
    
    print(f"\nOptimal parameters: θ₀={theta_optimal[0][0]:.4f}, θ₁={theta_optimal[1][0]:.4f}")
    print(f"Converged in {gd.convergence_iterations} iterations")
    print(f"Final cost: {gd.history['cost'][-1]:.6f}")
    
    # Visualize results
    plot_cost_convergence(gd.history)
    plot_parameter_path(gd.history['parameters'], true_theta)
    
    # ---------------------------------------------------------
    # 2. Compare Learning Rates
    # ---------------------------------------------------------
    print("\n" + "-"*40)
    print("2. Effect of Different Learning Rates")
    print("-"*40)
    
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx, lr in enumerate(learning_rates):
        gd_lr = GradientDescent(learning_rate=lr, max_iterations=100)
        gd_lr.fit(X, y, verbose=False)
        
        axes[idx].plot(gd_lr.history['cost'])
        axes[idx].set_title(f'Learning Rate = {lr}')
        axes[idx].set_xlabel('Iteration')
        axes[idx].set_ylabel('Cost')
        axes[idx].grid(True, alpha=0.3)
        
        if gd_lr.history['cost'][-1] < 100:
            axes[idx].axhline(y=gd_lr.history['cost'][-1], 
                            color='red', linestyle='--', alpha=0.5)
    
    plt.suptitle('Comparison of Learning Rates', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # ---------------------------------------------------------
    # 3. Compare Gradient Descent Variants
    # ---------------------------------------------------------
    print("\n" + "-"*40)
    print("3. Compare GD Variants")
    print("-"*40)
    
    # Batch GD
    gd_batch = GradientDescent(learning_rate=0.05, max_iterations=200)
    gd_batch.fit(X, y, verbose=False)
    
    # SGD (using our implementation)
    theta_sgd, costs_sgd = stochastic_gradient_descent(X, y, learning_rate=0.01, epochs=50)
    
    # Mini-batch GD
    theta_mbgd, costs_mbgd = mini_batch_gradient_descent(X, y, learning_rate=0.01, batch_size=32, epochs=100)
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Ensure costs are 1D arrays
    batch_costs = np.array(gd_batch.history['cost'])
    sgd_costs = np.array(costs_sgd).flatten()
    mbgd_costs = np.array(costs_mbgd).flatten()
    
    ax.plot(batch_costs, label='Batch GD', linewidth=2)
    ax.plot(range(0, len(sgd_costs)), sgd_costs, label='SGD', linewidth=2, alpha=0.7)
    ax.plot(range(0, len(mbgd_costs)), mbgd_costs, label='Mini-batch GD', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Iteration/Epoch')
    ax.set_ylabel('Cost')
    ax.set_title('Comparison of Gradient Descent Variants')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()
    
    # Print final results
    print(f"\nFinal Costs after convergence:")
    print(f"  Batch GD:     {batch_costs[-1]:.6f}")
    print(f"  SGD:          {sgd_costs[-1]:.6f}")
    print(f"  Mini-batch GD: {mbgd_costs[-1]:.6f}")
    
    return gd


# ============================================
# PART 5: INTERACTIVE VISUALIZATION (OPTIONAL)
# ============================================

def interactive_gradient_descent_demo():
    """
    Create an interactive demonstration of gradient descent
    (Requires ipywidgets - uncomment and run in Jupyter)
    """
    try:
        from IPython.display import display, HTML
        import ipywidgets as widgets
        
        # Create sample data
        X, y, true_theta = create_sample_data(n_samples=100, noise=0.3)
        
        def update_plot(learning_rate, iterations):
            """Update the plot based on slider values"""
            gd = GradientDescent(learning_rate=learning_rate, max_iterations=iterations)
            gd.fit(X, y, verbose=False)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Cost convergence
            axes[0].plot(gd.history['cost'])
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel('Cost')
            axes[0].set_title(f'Cost Convergence\n(LR={learning_rate}, Iter={iterations})')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_yscale('log')
            
            # Parameter trajectory (if 2D)
            theta_hist = np.array(gd.history['parameters'])
            if theta_hist.shape[1] == 2:
                axes[1].plot(theta_hist[:, 0], theta_hist[:, 1], 'b-', alpha=0.6)
                axes[1].scatter(theta_hist[:, 0], theta_hist[:, 1], 
                              c=range(len(theta_hist)), cmap='viridis', 
                              s=20, alpha=0.7)
                axes[1].scatter(theta_hist[0, 0], theta_hist[0, 1], 
                              color='red', s=100, marker='o', label='Start')
                axes[1].scatter(theta_hist[-1, 0], theta_hist[-1, 1], 
                              color='green', s=100, marker='*', label='End')
                axes[1].set_xlabel('θ₀')
                axes[1].set_ylabel('θ₁')
                axes[1].set_title('Parameter Trajectory')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            print(f"Final parameters: θ₀={theta_hist[-1, 0]:.4f}, θ₁={theta_hist[-1, 1]:.4f}")
            print(f"Final cost: {gd.history['cost'][-1]:.6f}")
        
        # Create interactive widgets
        lr_slider = widgets.FloatSlider(
            value=0.1, min=0.001, max=1.0, step=0.001,
            description='Learning Rate:',
            continuous_update=False
        )
        
        iter_slider = widgets.IntSlider(
            value=100, min=10, max=500, step=10,
            description='Iterations:',
            continuous_update=False
        )
        
        widgets.interactive(update_plot, learning_rate=lr_slider, iterations=iter_slider)
        
    except ImportError:
        print("Interactive demo requires ipywidgets. Install with: pip install ipywidgets")


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Run the demonstration
    gd_model = demonstrate_gradient_descent()
    
    print("\n" + "="*60)
    print("GRADIENT DESCENT EXPLANATION")
    print("="*60)
    print("""
    Gradient Descent Algorithm Flow:
    ================================
    
    1. Initialize parameters (θ) randomly
    2. For each iteration:
       a. Compute predictions: ŷ = X·θ
       b. Calculate error: error = ŷ - y
       c. Compute gradient: ∇J = (1/m)·Xᵀ·error
       d. Update parameters: θ = θ - α·∇J
       e. Check convergence
    
    Key Concepts:
    -------------
    • Learning Rate (α): Controls step size
    • Gradient: Direction of steepest ascent
    • Cost Function: Measures model performance
    • Convergence: When cost stops decreasing
    
    Variants:
    ---------
    • Batch GD: Uses all data for gradient
    • Stochastic GD: Uses 1 sample at a time
    • Mini-batch GD: Uses small batch of samples
    """)
    
    print("\nTo run the interactive demo (in Jupyter):")
    print("  interactive_gradient_descent_demo()")