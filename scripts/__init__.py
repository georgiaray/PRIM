"""
Analysis Scripts Module

This module contains separate scripts for different analysis methods:
- Variable selection
- PRIM analysis
- SHAP analysis  
- Random Forest analysis
- PCA and Clustering analysis
"""

from .prim_analysis import (
    run_prim_analysis,
    assess_prim_results,
    rank_prim_variables,
    get_restricted_variables
)

from .shap_analysis import (
    run_shap_analysis,
    count_variables_to_reach_threshold,
    plot_shap_summary
)

from .random_forest_analysis import (
    run_random_forest,
    get_cluster_variable_mapping
)

from .pca_clustering_analysis import (
    apply_pca,
    plot_3d_pca_projection,
    plot_pca_pairplot,
    plot_cluster_boxplots,
    plot_combined_cluster_boxplots,
    scale_and_impute
)

from .variable_selection import (
    coverage_for_variable,
    variable_coverage_table,
    assign_sector,
    build_correlation_matrix,
    sample_variables,
    expand_variables,
    is_below_correlation_threshold,
    count_valid_scenarios,
    select_variable_combinations,
    make_heat_map,
    save_filtered_scenarios
)

__all__ = [
    # Variable Selection
    'coverage_for_variable',
    'variable_coverage_table',
    'assign_sector',
    'build_correlation_matrix',
    'sample_variables',
    'expand_variables',
    'is_below_correlation_threshold',
    'count_valid_scenarios',
    'select_variable_combinations',
    'make_heat_map',
    'save_filtered_scenarios',
    # PRIM
    'run_prim_analysis',
    'assess_prim_results',
    'rank_prim_variables',
    'get_restricted_variables',
    # SHAP
    'run_shap_analysis',
    'count_variables_to_reach_threshold',
    'plot_shap_summary',
    # Random Forest
    'run_random_forest',
    'get_cluster_variable_mapping',
    # PCA/Clustering
    'apply_pca',
    'plot_3d_pca_projection',
    'plot_pca_pairplot',
    'plot_cluster_boxplots',
    'plot_combined_cluster_boxplots',
    'scale_and_impute',
]
