"""
Random Forest Analysis Script

This script performs Random Forest analysis for feature importance
and cluster prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestClassifier


def run_random_forest(X_df, labels, n_estimators=200, random_state=42, plot=True):
    """
    Run Random Forest classifier and extract feature importances.
    
    Parameters:
    -----------
    X_df : pd.DataFrame
        Feature matrix
    labels : pd.Series or np.ndarray
        Target labels
    n_estimators : int, default=200
        Number of trees in the forest
    random_state : int, default=42
        Random seed for reproducibility
    plot : bool, default=True
        Whether to plot feature importances
        
    Returns:
    --------
    pd.Series
        Feature importances sorted in descending order
    """
    print("=" * 60)
    print("Running Random Forest Analysis")
    print("=" * 60)
    
    # Train Random Forest
    print(f"\nTraining Random Forest with {n_estimators} trees...")
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_df, labels)
    
    # Extract importances
    importances = pd.Series(rf.feature_importances_, index=X_df.columns)
    top_importances = importances.sort_values(ascending=False)
    
    if plot:
        print("\nPlotting feature importances...")
        top_importances.plot(kind='barh', title='Top Variables Predicting Clusters')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    print("\nTop 10 Most Important Variables:")
    print(top_importances.head(10))
    
    return top_importances


def get_cluster_variable_mapping(df_clustered, cluster_to_bucket, top_n=10, n_estimators=200, random_state=42):
    """
    Map variables to clusters using Random Forest importance.
    
    Parameters:
    -----------
    df_clustered : pd.DataFrame
        DataFrame with cluster labels
    cluster_to_bucket : dict
        Mapping from cluster IDs to temperature buckets
    top_n : int, default=10
        Number of top variables to return per cluster
    n_estimators : int, default=200
        Number of trees in Random Forest
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with variables, clusters, buckets, and importance scores
    """
    print("=" * 60)
    print("Computing Cluster-Variable Mapping")
    print("=" * 60)
    
    results = []
    
    X = df_clustered.drop(columns=['model', 'scenario', 'Category', 'temp_bucket', 'cluster'], errors='ignore')
    
    for cluster_id in sorted(df_clustered['cluster'].unique()):
        print(f"\nAnalyzing cluster {cluster_id}...")
        y = (df_clustered['cluster'] == cluster_id).astype(int)
        
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        rf.fit(X, y)
        
        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(top_n)
        
        bucket = cluster_to_bucket.get(cluster_id, "Unknown")
        
        for var, score in importances.items():
            results.append({
                "Variable": var,
                "Cluster": cluster_id,
                "Bucket (from Cluster)": bucket,
                "Cluster RF Importance": score
            })
    
    result_df = pd.DataFrame(results)
    print(f"\nGenerated mapping for {len(result_df)} variable-cluster pairs.")
    
    return result_df


def main():
    """
    Example usage of Random Forest analysis.
    """
    print("Random Forest Analysis Script")
    print("=" * 60)
    print("\nThis script provides functions for Random Forest analysis.")
    print("Import this module and use the functions with your data.")
    print("\nExample usage:")
    print("  from scripts.random_forest_analysis import run_random_forest")
    print("  importances = run_random_forest(X_df, labels)")
    print("=" * 60)


if __name__ == "__main__":
    main()
