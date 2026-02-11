"""
Variable Selection Module

This module contains functions for selecting variables for PRIM analysis.
The selection process considers:
- Coverage across scenarios and years
- Correlation between variables
- Sector-based organization
- Completeness thresholds

Current approach:
- Multi-year support (2030, 2050)
- Systematic sampling of combinations
- Automated sector assignment via NLP/keywords
- Configurable thresholds
"""

import numpy as np
import pandas as pd
import random
import re
from collections import defaultdict


def coverage_for_variable(df_pivot, variable, years):
    """
    Check the coverage of a variable across specified years in a DataFrame.
    
    Parameters:
    -----------
    df_pivot : pd.DataFrame
        DataFrame with MultiIndex columns (variable, year)
    variable : str
        The variable to check coverage for
    years : list
        List of years to check coverage for
        
    Returns:
    --------
    int
        Number of complete rows for the specified variable across the years
    """
    # Build column list for the variable across years
    cols = [(variable, year) for year in years]

    # Check that both columns exist
    if not all(col in df_pivot.columns for col in cols):
        missing = [str(col) for col in cols if col not in df_pivot.columns]
        print(f"Missing columns for variable '{variable}': {missing}")
        return 0

    # Subset and count complete rows
    complete_rows = df_pivot[cols].dropna()
    return len(complete_rows)


def variable_coverage_table(df_pivot, years):
    """
    Generate a coverage table for all variables in the DataFrame across specified years.
    
    Parameters:
    -----------
    df_pivot : pd.DataFrame
        DataFrame with MultiIndex columns (variable, year)
    years : list
        List of years to check coverage for
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with variable names and their coverage counts, sorted ascending
    """
    # Extract base variable names
    base_vars = sorted(set(var for var, year in df_pivot.columns if year in years))

    coverage_counts = []

    for var in base_vars:
        cols = [(var, year) for year in years]
        if all(col in df_pivot.columns for col in cols):
            count = df_pivot[cols].dropna().shape[0]
        else:
            count = 0
        coverage_counts.append({"Variable": var, "Coverage (Scenarios)": count})

    # Convert to DataFrame and sort
    coverage_df = pd.DataFrame(coverage_counts)
    coverage_df = coverage_df.sort_values("Coverage (Scenarios)", ascending=True).reset_index(drop=True)

    return coverage_df


def assign_sector(column_name, sector_assignment):
    """
    Assign a variable to a sector based on a sector assignment DataFrame.
    
    Parameters:
    -----------
    column_name : str
        Variable name to assign
    sector_assignment : pd.DataFrame
        DataFrame with 'Variable' and 'Sector' columns
        
    Returns:
    --------
    str
        Sector name or "Other" if no match
    """
    sector_map = dict(zip(sector_assignment['Variable'], sector_assignment['Sector']))
    
    if column_name in sector_map:
        return sector_map[column_name]
    
    print(f"Unassigned column → '{column_name}' → sector: 'Other'")
    return "Other"


def build_correlation_matrix(df_pivot, year=2030):
    """
    Build correlation matrix for a specific year.
    
    Parameters:
    -----------
    df_pivot : pd.DataFrame
        DataFrame with MultiIndex columns (variable, year)
    year : int, default=2030
        Year to build correlation matrix for
        
    Returns:
    --------
    pd.DataFrame
        Absolute correlation matrix
    """
    df_year = df_pivot.loc[:, df_pivot.columns.get_level_values(1) == year]
    corr_matrix = df_year.corr().abs()
    return corr_matrix


def sample_variables(sector_vars, min_vars=2, max_vars=5):
    """
    Sample initial variables from each sector.
    
    Parameters:
    -----------
    sector_vars : dict
        Dictionary mapping sector names to lists of variables
    min_vars : int, default=2
        Minimum variables to sample per sector
    max_vars : int, default=5
        Maximum variables per sector (not used in initial sampling)
        
    Returns:
    --------
    tuple
        (selected_vars, sector_pool) where:
        - selected_vars: list of initially selected variables
        - sector_pool: dict of remaining variables per sector
    """
    selected = []
    sector_pool = {}
    
    for sector, vars in sector_vars.items():
        if len(vars) < min_vars:
            return None, None 
        base = random.sample(vars, min_vars)
        selected.extend(base)
        remaining = list(set(vars) - set(base))
        random.shuffle(remaining)
        sector_pool[sector] = remaining
    
    return selected, sector_pool


def expand_variables(selected, sector_pool, sector_vars, corr_matrix, max_vars=5, corr_threshold=0.9):
    """
    Expand variable combinations while maintaining correlation constraints.
    
    Parameters:
    -----------
    selected : list
        Initially selected variables
    sector_pool : dict
        Remaining variables per sector
    sector_vars : dict
        Full mapping of sectors to variables
    corr_matrix : pd.DataFrame
        Correlation matrix
    max_vars : int, default=5
        Maximum variables per sector
    corr_threshold : float, default=0.9
        Maximum allowed correlation
        
    Returns:
    --------
    list
        Expanded variable combination
    """
    combo = list(selected)

    for sector, remaining in sector_pool.items():
        current = [v for v in combo if v in sector_vars[sector]]
        while len(current) < max_vars and remaining:
            v = remaining.pop()
            test_combo = combo + [v]
            sub_corr = corr_matrix.loc[test_combo, test_combo].fillna(1).copy()
            # Set diagonal to 0 using DataFrame operations instead of numpy
            sub_corr_np = sub_corr.to_numpy().copy()
            np.fill_diagonal(sub_corr_np, 0)
            if sub_corr_np.max() <= corr_threshold:
                combo.append(v)
                current.append(v)
    
    return combo


def is_below_correlation_threshold(combo, corr_matrix, threshold=0.9):
    """
    Check if variable combination meets correlation threshold.
    
    Parameters:
    -----------
    combo : list
        List of variable names
    corr_matrix : pd.DataFrame
        Correlation matrix
    threshold : float, default=0.9
        Maximum allowed correlation
        
    Returns:
    --------
    tuple
        (bool, float) - (passes threshold, max correlation)
    """
    sub_corr = corr_matrix.loc[combo, combo].fillna(1).copy()
    # Set diagonal to 0 using numpy array copy
    sub_corr_np = sub_corr.to_numpy().copy()
    np.fill_diagonal(sub_corr_np, 0)
    max_corr = sub_corr_np.max()
    return max_corr <= threshold, max_corr


def count_valid_scenarios(df_pivot, combo, years, threshold=0.8):
    """
    Count scenarios meeting completeness threshold for variable combination.
    
    Parameters:
    -----------
    df_pivot : pd.DataFrame
        DataFrame with MultiIndex columns (variable, year)
    combo : list
        List of variable names
    years : list
        List of years to check
    threshold : float, default=0.8
        Completeness threshold (fraction of non-null values required)
        
    Returns:
    --------
    int
        Number of valid scenarios
    """
    candidate_cols = [(v, y) for v in combo for y in years if (v, y) in df_pivot.columns]
    if len(candidate_cols) < len(combo) * len(years):
        return 0

    subset = df_pivot[candidate_cols].copy()
    completeness = subset.notna().sum(axis=1) >= int(len(candidate_cols) * threshold)
    valid_scenarios = subset[completeness]
    return valid_scenarios.shape[0]


def select_variable_combinations(
    df_pivot,
    sector_vars,
    years=[2030, 2050],
    min_vars=2,
    max_vars=5,
    completeness_threshold=0.8,
    corr_threshold=0.9,
    num_samples=1000
):
    """
    Select variable combinations through systematic sampling.
    
    This is the main function for variable selection. It:
    1. Samples initial variables from each sector
    2. Expands combinations while checking correlation
    3. Filters by completeness threshold
    4. Returns ranked combinations
    
    Parameters:
    -----------
    df_pivot : pd.DataFrame
        DataFrame with MultiIndex columns (variable, year)
    sector_vars : dict
        Dictionary mapping sector names to lists of variables
    years : list, default=[2030, 2050]
        Years to consider
    min_vars : int, default=2
        Minimum variables per sector
    max_vars : int, default=5
        Maximum variables per sector
    completeness_threshold : float, default=0.8
        Required fraction of non-null values
    corr_threshold : float, default=0.9
        Maximum allowed correlation between variables
    num_samples : int, default=1000
        Number of random combinations to sample
        
    Returns:
    --------
    tuple
        (results, sector_info) where:
        - results: list of dicts with 'variables', 'n_scenarios', 'max_corr'
        - sector_info: list of (variable, sector) tuples (from last iteration)
    """
    variable_to_sector = {
        var: sector for sector, vars_list in sector_vars.items() for var in vars_list
    }
    corr_matrix = build_correlation_matrix(df_pivot, year=2030)
    results = []

    for _ in range(num_samples):
        selected, sector_pool = sample_variables(sector_vars, min_vars, max_vars)
        if selected is None:
            continue

        combo = expand_variables(selected, sector_pool, sector_vars, corr_matrix, max_vars, corr_threshold)
        passed, max_corr = is_below_correlation_threshold(combo, corr_matrix, corr_threshold)
        if not passed:
            continue

        n_scenarios = count_valid_scenarios(df_pivot, combo, years, threshold=completeness_threshold)
        if n_scenarios == 0:
            continue

        sector_info = [(var, variable_to_sector.get(var, 'Unknown')) for var in combo]
        results.append({
            "variables": combo,
            "n_scenarios": n_scenarios,
            "max_corr": max_corr
        })

    results = sorted(results, key=lambda x: -x["n_scenarios"])

    for i, res in enumerate(results[:10]):
        print(f"Combo {i+1}: {res['n_scenarios']} scenarios | Max corr: {res['max_corr']:.2f}")
        print(f"Variables: {res['variables']}\n")

    return results, sector_info


def make_heat_map(df_pivot, combo_vars, years=[2030], threshold=0.8, plot=True):
    """
    Visualize correlation across selected variables for specified year(s).
    
    Parameters:
    -----------
    df_pivot : pd.DataFrame
        DataFrame with MultiIndex columns (variable, year)
    combo_vars : list
        List of variable names to visualize
    years : list, default=[2030]
        Years to include
    threshold : float, default=0.8
        Completeness threshold (not used for visualization, kept for API consistency)
    plot : bool, default=True
        Whether to display the plot
        
    Returns:
    --------
    tuple
        (top_10_correlations, df_subset) where:
        - top_10_correlations: Series of top 10 correlated pairs
        - df_subset: DataFrame with selected variables
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Step 1: Build multi-index column list
    selected_cols = [(v, y) for v in combo_vars for y in years if (v, y) in df_pivot.columns]
    if len(selected_cols) < len(combo_vars) * len(years):
        missing = set(combo_vars) - {v for v, y in selected_cols}
        print(f"Warning: missing columns for variables: {missing}")

    # Step 2: Subset without dropping incomplete rows
    df_subset = df_pivot[selected_cols].copy()
    df_subset.columns = [f"{var} ({year})" for var, year in df_subset.columns]
    df_subset = df_subset.reset_index()

    # Step 3: Select 2030-only columns for correlation
    corr_cols = [col for col in df_subset.columns if "2030" in col]
    corr = df_subset[corr_cols].corr()

    # Step 4: Plot
    if plot:
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr,
            cmap="coolwarm",
            annot=True,
            fmt=".2f",
            center=0,
            mask=corr.isnull()
        )
        plt.title(f"Correlation Heatmap for {len(combo_vars)} Variables (2030 only)")
        plt.tight_layout()
        plt.show()

    # Step 5: Top correlations
    corr_flat = corr.abs().unstack().dropna()
    corr_flat = corr_flat[corr_flat < 1.0]
    top_10 = corr_flat.sort_values(ascending=False).head(10)

    print("\nTop 10 Most Correlated Pairs:")
    for (v1, v2), val in top_10.items():
        print(f"{v1} & {v2}: {val:.2f}")

    return top_10, df_subset


def save_filtered_scenarios(df_pivot, combo_vars, years=[2030, 2050], threshold=0.8, save_path=None):
    """
    Filter df_pivot to only include scenarios with >= threshold completeness
    for the given combo_vars and years. Optionally saves to CSV.
    
    Parameters:
    -----------
    df_pivot : pd.DataFrame
        Pivoted scenario data with (variable, year) columns
    combo_vars : list
        Variables to include
    years : list, default=[2030, 2050]
        Years to include
    threshold : float, default=0.8
        Required fraction of non-null values
    save_path : str or None, default=None
        Path to save filtered CSV. If None, only returns.
        
    Returns:
    --------
    pd.DataFrame
        Filtered scenarios with flattened column names
    """
    # Step 1: Build list of columns
    cols = [(v, y) for v in combo_vars for y in years if (v, y) in df_pivot.columns]
    if not cols:
        raise ValueError("No valid variable-year combinations found in df_pivot.")

    # Step 2: Filter rows with sufficient completeness
    subset = df_pivot[cols].copy()
    completeness = subset.notna().sum(axis=1) >= int(len(cols) * threshold)
    filtered = subset[completeness].copy()

    if filtered.empty:
        print("No scenarios passed the completeness threshold.")
        return pd.DataFrame()

    # Step 3: Flatten column MultiIndex
    filtered.columns = [f"{v} ({y})" for v, y in filtered.columns]
    filtered = filtered.reset_index()

    # Step 4: Save or return
    if save_path:
        filtered.to_csv(save_path, index=False)
        print(f"Filtered DataFrame saved to: {save_path}")

    return filtered
