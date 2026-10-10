import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def create_directory():
    """Create directory for datasets if it doesn't exist"""
    if not os.path.exists('ml_datasets'):
        os.makedirs('ml_datasets')
    return 'ml_datasets'

def generate_classification_dataset(n_samples=1000, n_features=10, n_classes=2, 
                                   dataset_name='classification'):
    """
    Generate a clean classification dataset
    
    Parameters:
    - n_samples: Number of samples
    - n_features: Number of features
    - n_classes: Number of classes
    - dataset_name: Name for the dataset
    """
    # Generate the dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.7),
        n_redundant=int(n_features * 0.2),
        n_classes=n_classes,
        n_clusters_per_class=min(2, n_classes),
        random_state=42,
        flip_y=0.01,  # Add 1% label noise
        class_sep=1.0  # Good separation between classes
    )
    
    # Create feature names
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add some categorical features for realism
    df['category_A'] = np.random.choice(['Type_X', 'Type_Y', 'Type_Z'], size=n_samples)
    df['category_B'] = np.random.choice(['Low', 'Medium', 'High'], size=n_samples)
    
    # Ensure no missing values
    assert df.isnull().sum().sum() == 0, "Dataset contains missing values!"
    
    # Scale numerical features
    numerical_cols = feature_names
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Save to CSV
    filepath = f'ml_datasets/{dataset_name}.csv'
    df.to_csv(filepath, index=False)
    
    print(f"Classification dataset saved: {filepath}")
    print(f"  Shape: {df.shape}")
    print(f"  Classes: {df['target'].nunique()}")
    print(f"  Class distribution:\n{df['target'].value_counts().to_string()}\n")
    
    return df

def generate_regression_dataset(n_samples=1000, n_features=10, 
                               dataset_name='regression'):
    """
    Generate a clean regression dataset
    
    Parameters:
    - n_samples: Number of samples
    - n_features: Number of features
    - dataset_name: Name for the dataset
    """
    # Generate the dataset
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.8),
        noise=10.0,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add some realistic features
    df['temperature'] = np.random.normal(25, 5, n_samples)
    df['humidity'] = np.random.uniform(30, 90, n_samples)
    df['day_of_week'] = np.random.randint(1, 8, n_samples)
    df['is_weekend'] = (df['day_of_week'] > 5).astype(int)
    
    # Ensure no missing values
    assert df.isnull().sum().sum() == 0, "Dataset contains missing values!"
    
    # Scale numerical features (excluding target)
    numerical_cols = [col for col in df.columns if col != 'target' and 
                     df[col].dtype in ['float64', 'int64']]
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Save to CSV
    filepath = f'ml_datasets/{dataset_name}.csv'
    df.to_csv(filepath, index=False)
    
    print(f"Regression dataset saved: {filepath}")
    print(f"  Shape: {df.shape}")
    print(f"  Target range: [{df['target'].min():.2f}, {df['target'].max():.2f}]")
    print(f"  Target mean: {df['target'].mean():.2f}")
    print(f"  Target std: {df['target'].std():.2f}\n")
    
    return df

def generate_binary_classification_dataset(n_samples=1000, n_features=8,
                                          dataset_name='binary_classification'):
    """Generate a clean binary classification dataset"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=6,
        n_redundant=1,
        n_classes=2,
        random_state=42,
        weights=[0.6, 0.4]  # Slight class imbalance
    )
    
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add categorical features
    df['gender'] = np.random.choice(['M', 'F'], size=n_samples)
    df['age_group'] = np.random.choice(['Young', 'Middle', 'Old'], size=n_samples)
    
    # Scale numerical features
    scaler = StandardScaler()
    df[feature_names] = scaler.fit_transform(df[feature_names])
    
    filepath = f'ml_datasets/{dataset_name}.csv'
    df.to_csv(filepath, index=False)
    
    print(f"Binary classification dataset saved: {filepath}")
    print(f"  Shape: {df.shape}")
    print(f"  Class distribution:\n{df['target'].value_counts().to_string()}\n")
    
    return df

def generate_multiclass_dataset(n_samples=1500, n_features=12, n_classes=4,
                               dataset_name='multiclass_classification'):
    """Generate a clean multiclass classification dataset"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=9,
        n_redundant=2,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=42
    )
    
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add some categorical features
    df['region'] = np.random.choice(['North', 'South', 'East', 'West'], size=n_samples)
    df['product_type'] = np.random.choice(['A', 'B', 'C'], size=n_samples)
    
    # Scale numerical features
    scaler = StandardScaler()
    df[feature_names] = scaler.fit_transform(df[feature_names])
    
    filepath = f'ml_datasets/{dataset_name}.csv'
    df.to_csv(filepath, index=False)
    
    print(f"Multiclass classification dataset saved: {filepath}")
    print(f"  Shape: {df.shape}")
    print(f"  Number of classes: {df['target'].nunique()}")
    print(f"  Class distribution:\n{df['target'].value_counts().to_string()}\n")
    
    return df

def generate_clustering_dataset(n_samples=1000, n_features=2, n_clusters=4,
                               dataset_name='clustering'):
    """Generate a dataset suitable for clustering (but can be used for classification)"""
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.0,
        random_state=42
    )
    
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['cluster'] = y
    
    # Scale features
    scaler = StandardScaler()
    df[feature_names] = scaler.fit_transform(df[feature_names])
    
    filepath = f'ml_datasets/{dataset_name}.csv'
    df.to_csv(filepath, index=False)
    
    print(f"Clustering dataset saved: {filepath}")
    print(f"  Shape: {df.shape}")
    print(f"  Number of clusters: {df['cluster'].nunique()}\n")
    
    return df

def generate_time_series_like_dataset(n_samples=1000, dataset_name='time_series'):
    """Generate a dataset with temporal patterns (can be used for regression)"""
    np.random.seed(42)
    
    # Create time-based features
    time_index = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Generate trend
    trend = np.linspace(100, 200, n_samples)
    
    # Generate seasonality
    seasonality = 20 * np.sin(2 * np.pi * np.arange(n_samples) / 365)
    
    # Generate noise
    noise = np.random.normal(0, 5, n_samples)
    
    # Target variable
    target = trend + seasonality + noise
    
    # Create additional features
    df = pd.DataFrame({
        'date': time_index,
        'year': time_index.year,
        'month': time_index.month,
        'day_of_week': time_index.dayofweek,
        'day_of_year': time_index.dayofyear,
        'quarter': time_index.quarter,
        'is_weekend': (time_index.dayofweek >= 5).astype(int),
        'temperature': np.random.normal(20, 10, n_samples),
        'humidity': np.random.uniform(40, 80, n_samples),
        'promotion': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
        'target': target
    })
    
    # Scale numerical features (excluding target and date)
    numerical_cols = ['temperature', 'humidity']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    filepath = f'ml_datasets/{dataset_name}.csv'
    df.to_csv(filepath, index=False)
    
    print(f"Time series dataset saved: {filepath}")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}\n")
    
    return df

def create_dataset_summary():
    """Create a summary file describing all datasets"""
    summary = """
    ============================================================
    MACHINE LEARNING DATASETS - SUMMARY
    ============================================================
    
    All datasets are:
    - Clean (no missing values)
    - Preprocessed (numerical features are scaled)
    - Ready for ML training and testing
    - Split into features and target (target column named 'target' or 'cluster')
    
    DATASET DESCRIPTIONS:
    ---------------------
    
    1. classification.csv
       - Binary classification dataset
       - 1000 samples, 10 numerical features
       - 2 categorical features
       - Target: 'target' (0 or 1)
       - Use for: Logistic Regression, SVM, Random Forest, etc.
    
    2. binary_classification.csv
       - Binary classification with class imbalance
       - 1000 samples, 8 numerical features
       - 2 categorical features
       - Target: 'target' (0 or 1)
       - Use for: Binary classification algorithms
    
    3. multiclass_classification.csv
       - Multiclass classification
       - 1500 samples, 12 numerical features
       - 2 categorical features
       - Target: 'target' (0, 1, 2, 3)
       - Use for: Multiclass classification, One-vs-Rest
    
    4. regression.csv
       - Regression dataset
       - 1000 samples, 10 numerical features
       - 4 additional realistic features
       - Target: 'target' (continuous)
       - Use for: Linear Regression, Ridge, Lasso, SVR, etc.
    
    5. time_series.csv
       - Time series regression dataset
       - 1000 samples with temporal features
       - Target: 'target' (continuous with trend/seasonality)
       - Use for: Time series regression, Feature engineering
    
    6. clustering.csv
       - Clustering dataset
       - 1000 samples, 2 numerical features
       - Target: 'cluster' (0, 1, 2, 3)
       - Use for: K-Means, DBSCAN, Hierarchical clustering
    
    ============================================================
    """
    
    with open('ml_datasets/README.txt', 'w') as f:
        f.write(summary)
    
    print(summary)

def verify_datasets():
    """Verify all datasets are clean and ready for ML"""
    print("="*60)
    print("VERIFYING ALL DATASETS")
    print("="*60)
    
    files = os.listdir('ml_datasets')
    csv_files = [f for f in files if f.endswith('.csv')]
    
    for file in csv_files:
        df = pd.read_csv(f'ml_datasets/{file}')
        print(f"\n📊 {file}")
        print(f"   Shape: {df.shape}")
        print(f"   Missing values: {df.isnull().sum().sum()}")
        print(f"   Data types: {df.dtypes.value_counts().to_dict()}")
        
        # Check for infinite values
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        print(f"   Infinite values: {inf_count}")
        
        # Check if target column exists
        target_cols = ['target', 'cluster']
        found_target = [col for col in target_cols if col in df.columns]
        if found_target:
            print(f"   Target column(s): {found_target}")
        else:
            print(f"   ⚠️ No target column found")

def main():
    """Main function to generate all datasets"""
    print("="*60)
    print("MACHINE LEARNING DATASET GENERATOR")
    print("="*60)
    print("\nGenerating clean datasets for ML algorithms...\n")
    
    # Create directory
    create_directory()
    
    # Generate different datasets
    print("Generating datasets...\n")
    
    # Classification datasets
    generate_classification_dataset(
        n_samples=1000, n_features=10, n_classes=2,
        dataset_name='classification'
    )
    
    generate_binary_classification_dataset(
        n_samples=1000, n_features=8,
        dataset_name='binary_classification'
    )
    
    generate_multiclass_dataset(
        n_samples=1500, n_features=12, n_classes=4,
        dataset_name='multiclass_classification'
    )
    
    # Regression datasets
    generate_regression_dataset(
        n_samples=1000, n_features=10,
        dataset_name='regression'
    )
    
    # Time series dataset
    generate_time_series_like_dataset(
        n_samples=1000,
        dataset_name='time_series'
    )
    
    # Clustering dataset (can also be used for classification)
    generate_clustering_dataset(
        n_samples=1000, n_features=2, n_clusters=4,
        dataset_name='clustering'
    )
    
    # Create summary
    create_dataset_summary()
    
    # Verify all datasets
    verify_datasets()
    
    print("\n" + "="*60)
    print("✅ ALL DATASETS GENERATED SUCCESSFULLY!")
    print("="*60)
    print("\nDatasets are saved in the 'ml_datasets' folder:")
    print("  📁 ml_datasets/")
    print("     ├── classification.csv")
    print("     ├── binary_classification.csv")
    print("     ├── multiclass_classification.csv")
    print("     ├── regression.csv")
    print("     ├── time_series.csv")
    print("     ├── clustering.csv")
    print("     └── README.txt")
    print("\nAll datasets are clean and ready for ML training/testing!")

if __name__ == "__main__":
    main()