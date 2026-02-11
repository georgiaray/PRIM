"""
PCA and Clustering Analysis Script

This script performs Principal Component Analysis (PCA) and clustering
to explore data structure and identify patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull


# Temperature bucket colors
temp_colors = {
    "1.5": "#D9420B",
    "2": "#035AA6",
    "above 2": "#338474",
    "failed-vetting": "#999999",
    "no-climate-assessment": "#CCCCCC"
}


def apply_pca(X_scaled, n_components=5, plot=True):
    """
    Apply Principal Component Analysis to scaled data.
    
    Parameters:
    -----------
    X_scaled : np.ndarray or pd.DataFrame
        Scaled feature matrix
    n_components : int, default=5
        Number of principal components
    plot : bool, default=True
        Whether to plot explained variance
        
    Returns:
    --------
    np.ndarray
        Transformed data in PCA space
    """
    print("=" * 60)
    print("Running PCA Analysis")
    print("=" * 60)
    
    print(f"\nFitting PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    if plot:
        print("\nPlotting explained variance...")
        plt.figure(figsize=(8, 4))
        plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("Explained Variance by PCA Components")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    print(f"\nExplained variance by component:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)")
    print(f"\nTotal explained variance: {np.sum(pca.explained_variance_ratio_):.4f} ({np.sum(pca.explained_variance_ratio_)*100:.2f}%)")
    
    return X_pca


def plot_3d_pca_projection(df_pca, X_pca):
    """
    Create 3D projection of PCA results with temperature buckets.
    
    Parameters:
    -----------
    df_pca : pd.DataFrame
        DataFrame with PCA components and temp_bucket column
    X_pca : np.ndarray
        PCA-transformed data
    """
    print("\nGenerating 3D PCA projection...")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for temp_label, hex_color in temp_colors.items():
        # Handle both old and new temp_bucket values for backward compatibility
        if temp_label == "2.0":
            mask = df_pca['temp_bucket'] == "2"
            display_label = "2.0°C"
        elif temp_label == "Above 2.0":
            mask = df_pca['temp_bucket'] == "above 2"
            display_label = "Above 2.0°C"
        else:
            mask = df_pca['temp_bucket'] == temp_label
            # Format label for display
            if temp_label == "1.5":
                display_label = "1.5°C"
            elif temp_label == "2":
                display_label = "2.0°C"
            elif temp_label == "above 2":
                display_label = "Above 2.0°C"
            else:
                display_label = temp_label.replace("-", " ").title()
        pts = X_pca[mask.values][:, 1:4]  # Project on PC2, PC3, PC4
        
        # Scatter points
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=hex_color, alpha=0.2, label=display_label)
        
        # Convex hull if enough points
        if pts.shape[0] >= 4:
            try:
                hull = ConvexHull(pts)
                faces = [pts[simplex] for simplex in hull.simplices]
                poly = Poly3DCollection(faces, alpha=0.15, facecolor=hex_color, edgecolor='none')
                ax.add_collection3d(poly)
            except Exception as e:
                print(f"Failed to draw hull for {temp_label}°C: {e}")
    
    # Axes labels and styling
    ax.set_title("3D PCA Projection with Temperature Buckets")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3", labelpad=-8)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    ax.legend(title="Temperature Bucket", loc="upper left")
    ax.view_init(elev=10, azim=-100)
    
    plt.tight_layout()
    plt.show()


def plot_pca_pairplot(df_pca, n_components=6):
    """
    Create pairplot of PCA components.
    
    Parameters:
    -----------
    df_pca : pd.DataFrame
        DataFrame with PCA components and Cluster column
    n_components : int, default=6
        Number of components to plot
    """
    print("\nGenerating PCA pairplot...")
    
    sns.pairplot(df_pca, vars=df_pca.columns[:n_components], hue="Cluster", palette='viridis')
    plt.suptitle("Pairwise PCA Component Projections", y=1.02)
    plt.show()


def plot_cluster_boxplots(df, labels, top_vars):
    """
    Plot boxplots for top variables by cluster.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with variables
    labels : pd.Series or np.ndarray
        Cluster labels
    top_vars : list
        List of top variable names to plot
    """
    print("\nGenerating cluster boxplots...")
    
    for var in top_vars:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=labels, y=df[var])
        plt.title(f"{var} by Cluster")
        plt.xlabel("Cluster")
        plt.ylabel(var)
        plt.tight_layout()
        plt.show()


def plot_combined_cluster_boxplots(df, cluster_labels, variables, palette='Set2'):
    """
    Plot combined boxplots for multiple variables by cluster.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with variables
    cluster_labels : pd.Series or np.ndarray
        Cluster labels
    variables : list
        List of variable names to plot
    palette : str, default='Set2'
        Color palette for plots
    """
    print("\nGenerating combined cluster boxplots...")
    
    cluster_to_bucket = {
        0: "1.5°C",
        1: "Above 1.5"
    }
    
    bucket_labels = pd.Series(cluster_labels).map(cluster_to_bucket)
    bucket_labels = bucket_labels.fillna("Unknown")
    df_plot = df[variables].copy()
    df_plot['Temp Category'] = bucket_labels
    
    num_vars = len(variables)
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), sharey=False)
    axes = axes.flatten()
    
    unique_buckets = sorted(bucket_labels.unique(), key=lambda x: str(x))
    color_palette = sns.color_palette(palette, n_colors=len(unique_buckets))
    color_dict = dict(zip(sorted(unique_buckets), color_palette)) if isinstance(palette, str) else palette
    
    for i, var in enumerate(variables):
        sns.boxplot(
            data=df_plot,
            x='Temp Category',
            y=var,
            hue='Temp Category',
            palette=color_dict,
            ax=axes[i],
            dodge=False,
            order=["1.5°C", "Above 1.5"]  
        )
        axes[i].set_title(var)
        axes[i].legend_.remove()
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Standardized Value" if i == 0 else "")
    
    plt.suptitle("Distributions of Top Variables by Temperature Category", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def scale_and_impute(X):
    """
    Scale and impute missing values in feature matrix.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
        
    Returns:
    --------
    np.ndarray
        Scaled and imputed data
    scaler : StandardScaler
        Fitted scaler object
    """
    print("\nScaling and imputing data...")
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    return X_scaled, scaler


def main():
    """
    Example usage of PCA and clustering analysis.
    """
    print("PCA and Clustering Analysis Script")
    print("=" * 60)
    print("\nThis script provides functions for PCA and clustering analysis.")
    print("Import this module and use the functions with your data.")
    print("\nExample usage:")
    print("  from scripts.pca_clustering_analysis import apply_pca")
    print("  X_pca = apply_pca(X_scaled, n_components=5)")
    print("=" * 60)


if __name__ == "__main__":
    main()
