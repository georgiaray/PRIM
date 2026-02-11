"""
PRIM (Patient Rule Induction Method) Analysis Script

This script performs PRIM analysis to identify interpretable rules (boxes) 
in variable space that distinguish temperature pathways.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ema_workbench.analysis.prim import Prim


def run_prim_analysis(X_scaled, y, threshold=0.5, visualize=True):
    """
    Run PRIM analysis on scaled features and binary target.
    
    Parameters:
    -----------
    X_scaled : pd.DataFrame or np.ndarray
        Scaled feature matrix
    y : pd.Series or np.ndarray
        Binary target variable
    threshold : float, default=0.5
        PRIM threshold parameter
    visualize : bool, default=True
        Whether to show PRIM visualizations
        
    Returns:
    --------
    box : PRIM box object
        The fitted PRIM box
    box_lims : pd.DataFrame
        Box limits for each variable
    stats : list
        PRIM statistics for each box
    """
    print("=" * 60)
    print("Running PRIM Analysis")
    print("=" * 60)
    
    # Fit PRIM model
    p = Prim(X_scaled, y, threshold=threshold)
    box = p.find_box()
    
    if visualize:
        print("\nShowing PRIM tradeoff plot...")
        box.show_tradeoff(annotated=True)
        plt.show()
        
        print("\nShowing PRIM pairs scatter plot...")
        box.show_pairs_scatter()
        plt.show()
    
    return box, box.box_lims[-1], p.stats


def assess_prim_results(box_stats):
    """
    Assess PRIM box statistics and compute metrics.
    
    Parameters:
    -----------
    box_stats : list
        PRIM statistics from p.stats
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with precision, recall, F1 score, and other metrics
    """
    results = []
    
    for i, stats in enumerate(box_stats):
        try:
            precision = stats['density']    # EMA calls this "mean" in summary
            recall = stats['coverage']
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                "box_index": i,
                "precision (density)": round(precision, 3),
                "recall (coverage)": round(recall, 3),
                "f1_score": round(f1, 3),
                "restricted_dims": stats.get('res_dim'),
                "mass": round(stats['mass'], 3)
            })
        except Exception as e:
            print(f"Warning: skipping box {i} due to error: {e}")
    
    return pd.DataFrame(results)


def rank_prim_variables(lims, restricted_vars):
    """
    Rank variables by their importance in PRIM box (based on range width).
    
    Parameters:
    -----------
    lims : pd.DataFrame
        Box limits from PRIM (box.box_lims[-1])
    restricted_vars : list
        List of variables that were actually restricted
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with variables ranked by box range width
    """
    importance = []
    
    for var, (low, high) in lims.items():
        if var in restricted_vars:
            range_width = np.inf if np.isinf(low) or np.isinf(high) else high - low
        else:
            low, high = -np.inf, np.inf
            range_width = np.inf
        importance.append((var, range_width))
    
    return pd.DataFrame(importance, columns=["Variable", "Box Range Width"]).sort_values("Box Range Width")


def get_restricted_variables(prim_box, X_scaled, tol=1e-6):
    """
    Identify variables that PRIM actually restricted in the selected box.
    
    Parameters:
    -----------
    prim_box : PRIM box object
        The fitted PRIM box
    X_scaled : pd.DataFrame
        Scaled feature matrix
    tol : float, default=1e-6
        Tolerance for comparing bounds
        
    Returns:
    --------
    list
        List of restricted variable names
    """
    lb = prim_box.box_lims[-1].loc[0] 
    ub = prim_box.box_lims[-1].loc[1] 
    
    restricted_vars = []
    
    for variable in lb.index:
        data_min = X_scaled[variable].min()
        data_max = X_scaled[variable].max()
        
        if not np.isclose(lb[variable], data_min, atol=tol) or not np.isclose(ub[variable], data_max, atol=tol):
            print(f"Variable '{variable}' is restricted to [{lb[variable]:.4f}, {ub[variable]:.4f}] "
                  f"(data range: [{data_min:.4f}, {data_max:.4f}])")
            restricted_vars.append(variable)
    
    return restricted_vars


def main():
    """
    Example usage of PRIM analysis.
    This function demonstrates how to use the PRIM analysis functions.
    """
    print("PRIM Analysis Script")
    print("=" * 60)
    print("\nThis script provides functions for PRIM analysis.")
    print("Import this module and use the functions with your data.")
    print("\nExample usage:")
    print("  from scripts.prim_analysis import run_prim_analysis")
    print("  box, lims, stats = run_prim_analysis(X_scaled, y)")
    print("=" * 60)


if __name__ == "__main__":
    main()
