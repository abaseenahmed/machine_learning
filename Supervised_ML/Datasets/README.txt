
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
    