"""
SHAP (SHapley Additive exPlanations) Analysis Script

This script performs SHAP analysis to understand feature importance
and model interpretability using XGBoost.
"""

import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import shap
from xgboost import XGBClassifier


def count_variables_to_reach_threshold(shap_importance, threshold=0.8):
    """
    Count how many variables are needed to reach a threshold of total importance.
    
    Parameters:
    -----------
    shap_importance : pd.DataFrame
        DataFrame with 'SHAP Importance' column
    threshold : float, default=0.8
        Cumulative importance threshold (e.g., 0.8 for 80%)
        
    Returns:
    --------
    int
        Number of variables needed to reach threshold
    """
    sorted_importance = shap_importance["SHAP Importance"].sort_values(ascending=False).reset_index(drop=True)
    cumulative = sorted_importance.cumsum()
    total = sorted_importance.sum()
    num_vars = (cumulative / total < threshold).sum() + 1
    return num_vars


def run_shap_analysis(X, y, test_size=0.3, random_state=42, threshold=0.8):
    """
    Run SHAP analysis using XGBoost classifier.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series or np.ndarray
        Target variable
    test_size : float, default=0.3
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
    threshold : float, default=0.8
        Threshold for cumulative importance calculation
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with variables and their SHAP importance scores
    """
    print("=" * 60)
    print("Running SHAP Analysis")
    print("=" * 60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Train XGBoost model
    print("\nTraining XGBoost classifier...")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = xgb.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    # Compute SHAP values
    print("\nComputing SHAP values...")
    explainer = shap.Explainer(xgb)
    shap_values = explainer(X)
    
    # Summarize SHAP importance
    shap_importance = pd.DataFrame({
        "Variable": X.columns,
        "SHAP Importance": np.abs(shap_values.values).mean(axis=0)
    }).sort_values("SHAP Importance", ascending=False)
    
    # Calculate how many variables account for threshold of importance
    top_k = count_variables_to_reach_threshold(shap_importance, threshold=threshold)
    print(f"\n{top_k} variables account for {threshold*100:.0f}% of total SHAP importance.")
    
    print("\nTop 10 Most Important Variables:")
    print(shap_importance.head(10).to_string(index=False))
    
    return shap_importance


def plot_shap_summary(shap_values, X, max_display=10):
    """
    Plot SHAP summary plots.
    
    Parameters:
    -----------
    shap_values : shap.Explanation
        SHAP values from explainer
    X : pd.DataFrame
        Feature matrix
    max_display : int, default=10
        Maximum number of features to display
    """
    print("\nGenerating SHAP summary plots...")
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    import matplotlib.pyplot as plt
    plt.show()


def main():
    """
    Example usage of SHAP analysis.
    """
    print("SHAP Analysis Script")
    print("=" * 60)
    print("\nThis script provides functions for SHAP analysis.")
    print("Import this module and use the functions with your data.")
    print("\nExample usage:")
    print("  from scripts.shap_analysis import run_shap_analysis")
    print("  shap_importance = run_shap_analysis(X, y)")
    print("=" * 60)


if __name__ == "__main__":
    main()
