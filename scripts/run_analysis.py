"""
Main Analysis Pipeline

This script orchestrates the complete PRIM analysis workflow:
1. Load and preprocess data
2. Variable selection (top 10 combinations)
3. For each combination: PRIM, SHAP, RF, PCA/Clustering analysis
4. Save all results and visualizations
5. Generate comprehensive report
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.variable_selection import (
    select_variable_combinations,
    save_filtered_scenarios,
    make_heat_map
)
from scripts.prim_analysis import (
    run_prim_analysis,
    assess_prim_results,
    rank_prim_variables,
    get_restricted_variables
)
from scripts.shap_analysis import run_shap_analysis
from scripts.random_forest_analysis import (
    run_random_forest,
    get_cluster_variable_mapping
)
from scripts.pca_clustering_analysis import (
    apply_pca,
    plot_3d_pca_projection,
    plot_pca_pairplot,
    scale_and_impute
)


class AnalysisPipeline:
    """Main analysis pipeline orchestrator."""
    
    def __init__(self, data_path, output_folder_name=None):
        """
        Initialize the analysis pipeline.
        
        Parameters:
        -----------
        data_path : str
            Path to the input data CSV file
        output_folder_name : str, optional
            Name for the output folder. If None, will prompt user.
        """
        self.data_path = data_path
        self.output_folder_name = output_folder_name or self._prompt_output_folder()
        self.output_base = os.path.join('outputs', self.output_folder_name)
        self.visualizations_dir = os.path.join(self.output_base, 'visualizations')
        self.runs_dir = os.path.join(self.output_base, 'runs')
        
        # Create output directories
        os.makedirs(self.output_base, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)
        os.makedirs(self.runs_dir, exist_ok=True)
        
        # Storage for results
        self.results_summary = []
        self.df_pivot = None
        self.sector_vars = None
        
    def _prompt_output_folder(self):
        """Prompt user for output folder name."""
        print("\n" + "="*60)
        print("PRIM Analysis Pipeline")
        print("="*60)
        folder_name = input("\nEnter a name for this analysis run: ").strip()
        if not folder_name:
            # Default to timestamp
            folder_name = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return folder_name
    
    def load_category_mapping(self, category_mapping_path):
        """
        Load Category mapping from external CSV or Excel file.
        
        Parameters:
        -----------
        category_mapping_path : str
            Path to CSV or Excel file (.xlsx) with model, scenario, and Category columns
            
        Returns:
        --------
        pd.DataFrame or None
            Category mapping DataFrame, or None if path not provided/invalid
        """
        if not category_mapping_path or not os.path.exists(category_mapping_path):
            return None
        
        print("\n" + "="*60)
        print("Loading Category Mapping")
        print("="*60)
        print(f"Loading from: {category_mapping_path}")
        
        # Support both CSV and Excel files
        file_ext = os.path.splitext(category_mapping_path)[1].lower()
        if file_ext in ['.xlsx', '.xls']:
            try:
                # Try to read from 'meta' sheet first (common in AR6 data), fallback to first sheet
                try:
                    category_df = pd.read_excel(category_mapping_path, sheet_name='meta')
                except (ValueError, KeyError):
                    # If 'meta' sheet doesn't exist, try 'meta_Ch3vetted_withclimate'
                    try:
                        category_df = pd.read_excel(category_mapping_path, sheet_name='meta_Ch3vetted_withclimate')
                    except (ValueError, KeyError):
                        # Fallback to first sheet
                        category_df = pd.read_excel(category_mapping_path, sheet_name=0)
            except ImportError:
                raise ImportError(
                    "openpyxl is required for Excel file support. Install it with:\n"
                    "  pip install openpyxl\n"
                    "Or convert the Excel file to CSV format."
                )
        else:
            category_df = pd.read_csv(category_mapping_path)
        
        # Normalize column names to lowercase for matching
        # Map common variations: Model -> model, Scenario -> scenario, Category -> Category
        column_mapping = {}
        for col in category_df.columns:
            col_lower = str(col).lower()
            if col_lower in ['model', 'models']:
                column_mapping[col] = 'model'
            elif col_lower in ['scenario', 'scenarios']:
                column_mapping[col] = 'scenario'
            elif col_lower in ['category', 'categories']:
                column_mapping[col] = 'Category'
        
        # Rename columns if needed
        if column_mapping:
            category_df = category_df.rename(columns=column_mapping)
        
        # Validate required columns (case-insensitive check)
        required_cols = ['model', 'scenario', 'Category']
        available_cols_lower = {str(col).lower(): col for col in category_df.columns}
        missing_cols = []
        for req_col in required_cols:
            if req_col.lower() not in available_cols_lower:
                missing_cols.append(req_col)
        
        if missing_cols:
            raise ValueError(
                f"Category mapping file must have columns: {required_cols}. Missing: {missing_cols}\n"
                f"Found columns: {list(category_df.columns)}\n"
                f"Note: Column names are case-insensitive (Model/model, Scenario/scenario are both accepted)"
            )
        
        # Select only the required columns
        category_df = category_df[required_cols].copy()
        
        print(f"Loaded {len(category_df)} category mappings")
        print(f"Unique Categories: {category_df['Category'].value_counts().to_dict()}")
        
        return category_df
    
    def load_data(self, category_mapping_path=None):
        """
        Load and prepare data.
        
        Parameters:
        -----------
        category_mapping_path : str, optional
            Path to CSV file with model, scenario, and Category columns to join
        """
        print("\n" + "="*60)
        print("Loading Data")
        print("="*60)
        print(f"Loading data from: {self.data_path}")
        
        # Check if file exists, provide helpful error if not
        if not os.path.exists(self.data_path):
            # Try relative to project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            alt_path = os.path.join(project_root, self.data_path)
            if os.path.exists(alt_path):
                self.data_path = alt_path
                print(f"Found file at: {self.data_path}")
            else:
                raise FileNotFoundError(
                    f"Data file not found: {self.data_path}\n"
                    f"Also tried: {alt_path}\n"
                    f"Make sure you're running from the PRIM project root directory.\n"
                    f"Use paths like 'data/combined_ar6_data.csv' not '../data/combined_ar6_data.csv'"
                )
        
        # Load data (support CSV, Parquet, and Pickle)
        file_ext = os.path.splitext(self.data_path)[1].lower()
        if file_ext == '.parquet':
            df = pd.read_parquet(self.data_path)
            print(f"Loaded pivoted data from Parquet file")
        elif file_ext == '.pkl' or file_ext == '.pickle':
            df = pd.read_pickle(self.data_path)
            print(f"Loaded pivoted data from Pickle file")
        else:
            df = pd.read_csv(self.data_path)
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        
        # Join Category mapping if provided and Category not already in data
        if category_mapping_path:
            # Check if Category is already in data (either in columns or index)
            has_category_in_cols = 'Category' in df.columns if not isinstance(df.columns, pd.MultiIndex) else False
            has_category_in_index = isinstance(df.index, pd.MultiIndex) and 'Category' in df.index.names
            
            if not has_category_in_cols and not has_category_in_index:
                category_df = self.load_category_mapping(category_mapping_path)
                if category_df is not None:
                    # For pivoted data, we need to reset index to join, then rebuild
                    if isinstance(df.columns, pd.MultiIndex):
                        print("Joining Category mapping with pivoted data...")
                        # Get original index names before reset
                        original_index_names = [name for name in df.index.names if name]
                        
                        # Store the MultiIndex columns structure (data columns only, excluding index columns)
                        # After reset_index(), index columns become regular columns with empty second level
                        # We need to preserve the data columns structure
                        data_columns = [col for col in df.columns if col[0] not in original_index_names]
                        
                        # Reset index completely to get regular columns
                        df_reset = df.reset_index()
                        
                        # Store the MultiIndex column structure to restore later
                        original_multiindex_cols = df_reset.columns.copy() if isinstance(df_reset.columns, pd.MultiIndex) else None
                        
                        # Convert MultiIndex columns to regular columns for merging
                        # This is necessary because pandas can't merge MultiIndex columns with regular columns
                        if isinstance(df_reset.columns, pd.MultiIndex):
                            # Flatten MultiIndex columns to regular string columns
                            new_cols = []
                            col_mapping = {}  # Map old MultiIndex col -> new string col name
                            for col in df_reset.columns:
                                if isinstance(col, tuple):
                                    # For index columns like ('model', ''), use just the first part
                                    if col[1] == '' or pd.isna(col[1]):
                                        new_col = str(col[0])
                                    else:
                                        # For data columns like ('Variable', 2030), create a string representation
                                        new_col = f"{col[0]} ({col[1]})"
                                    col_mapping[col] = new_col
                                    new_cols.append(new_col)
                                else:
                                    new_cols.append(str(col))
                            
                            # Rename columns to regular strings
                            df_reset.columns = new_cols
                        
                        # Ensure category_df columns are regular (not MultiIndex) and lowercase
                        category_subset = category_df[['model', 'scenario', 'Category']].copy()
                        category_subset.columns = [str(col).lower() if col in ['model', 'scenario'] else str(col) 
                                                  for col in category_subset.columns]
                        
                        # Merge with category mapping using regular columns
                        merge_keys = ['model', 'scenario']
                        df_reset = df_reset.merge(
                            category_subset, 
                            on=merge_keys,
                            how='left'
                        )
                        
                        # Add Category to index if not already there
                        if 'Category' not in original_index_names:
                            original_index_names.append('Category')
                        
                        # Set index back - use only the columns that exist
                        available_index_cols = [col for col in original_index_names if col in df_reset.columns]
                        df = df_reset.set_index(available_index_cols)
                        
                        # Restore MultiIndex columns if they were converted to regular columns
                        # The columns should now be in format "Variable (2030)" - convert back to MultiIndex
                        if not isinstance(df.columns, pd.MultiIndex) and original_multiindex_cols is not None:
                            # Check if columns match the pattern "Variable (Year)"
                            new_cols = []
                            for col in df.columns:
                                if '(' in str(col) and ')' in str(col):
                                    # Parse "Variable (Year)" format
                                    parts = str(col).rsplit('(', 1)
                                    if len(parts) == 2:
                                        var_name = parts[0].strip()
                                        year_str = parts[1].rstrip(')').strip()
                                        try:
                                            year = int(year_str)
                                            new_cols.append((var_name, year))
                                        except ValueError:
                                            new_cols.append(col)
                                else:
                                    new_cols.append(col)
                            
                            # Only convert to MultiIndex if we have tuple columns
                            if any(isinstance(c, tuple) for c in new_cols):
                                df.columns = pd.MultiIndex.from_tuples(new_cols)
                        
                        print(f"  Joined Category: {df_reset['Category'].value_counts().to_dict() if 'Category' in df_reset.columns else 'Failed'}")
                        print(f"  Columns are MultiIndex: {isinstance(df.columns, pd.MultiIndex)}")
                    else:
                        print("Joining Category mapping with data...")
                        # For non-pivoted data, merge directly
                        df = df.merge(
                            category_df[['model', 'scenario', 'Category']].copy(), 
                            on=['model', 'scenario'], 
                            how='left'
                        )
                        print(f"  Joined Category: {df['Category'].value_counts().to_dict() if 'Category' in df.columns else 'Failed'}")
        
        # Check if data is already pivoted (MultiIndex columns)
        if isinstance(df.columns, pd.MultiIndex):
            self.df_pivot = df
            
            # Create temp_bucket from Category if Category is in index but temp_bucket is not
            if isinstance(self.df_pivot.index, pd.MultiIndex):
                index_names = list(self.df_pivot.index.names)
                if 'Category' in index_names and 'temp_bucket' not in index_names:
                    print("Creating temp_bucket from Category (in index)...")
                    df_reset = self.df_pivot.reset_index()
                    
                    def categorize_temp_bucket(row):
                        """Convert Category to temperature bucket."""
                        if pd.isna(row.get('Category', None)):
                            return 'Unknown'
                        cat = str(row['Category']).upper()
                        if cat in ['C1', 'C2']:
                            return '1.5'
                        elif cat in ['C3', 'C4']:
                            return '2.0'
                        else:
                            return 'Above 2.0'
                    
                    df_reset['temp_bucket'] = df_reset.apply(categorize_temp_bucket, axis=1)
                    
                    # Rebuild index - replace Category with temp_bucket
                    new_index_cols = [col for col in index_names if col != 'Category'] + ['temp_bucket']
                    self.df_pivot = df_reset.set_index(new_index_cols)
                    print(f"  Created temp_bucket: {df_reset['temp_bucket'].value_counts().to_dict()}")
            
            print("Data is already in pivoted format (MultiIndex columns)")
        else:
            # Check if data is in long format (has 'variable' and 'year' columns)
            if 'variable' in df.columns and 'year' in df.columns:
                print("Detected long format data. Pivoting to wide format...")
                
                # Join Category mapping if provided and Category not already in data
                if category_mapping_path and 'Category' not in df.columns:
                    category_df = self.load_category_mapping(category_mapping_path)
                    if category_df is not None:
                        print("Joining Category mapping with data...")
                        df = df.merge(category_df[['model', 'scenario', 'Category']], 
                                     on=['model', 'scenario'], 
                                     how='left')
                        print(f"  Joined Category: {df['Category'].value_counts().to_dict() if 'Category' in df.columns else 'Failed'}")
                
                # Pivot: variable and year become MultiIndex columns
                # Keep model, scenario, region, Category/temp_bucket as index
                id_cols = ['model', 'scenario']
                if 'region' in df.columns:
                    id_cols.append('region')
                if 'Category' in df.columns:
                    id_cols.append('Category')
                if 'temp_bucket' in df.columns:
                    id_cols.append('temp_bucket')
                
                # Filter to only columns that exist
                id_cols = [col for col in id_cols if col in df.columns]
                
                # Create temp_bucket from Category before pivoting if Category exists
                if 'Category' in df.columns and 'temp_bucket' not in df.columns:
                    print("Creating temp_bucket from Category column...")
                    def categorize_temp_bucket(row):
                        """Convert Category to temperature bucket."""
                        if pd.isna(row.get('Category', None)):
                            return 'Unknown'
                        cat = str(row['Category']).upper()
                        if cat in ['C1', 'C2']:
                            return '1.5'
                        elif cat in ['C3', 'C4']:
                            return '2.0'
                        else:
                            return 'Above 2.0'
                    
                    df['temp_bucket'] = df.apply(categorize_temp_bucket, axis=1)
                    print(f"  Created temp_bucket: {df['temp_bucket'].value_counts().to_dict()}")
                    
                    # Add temp_bucket to id_cols and remove Category
                    if 'Category' in id_cols:
                        id_cols.remove('Category')
                    if 'temp_bucket' not in id_cols:
                        id_cols.append('temp_bucket')
                
                # Pivot the data
                self.df_pivot = df.pivot_table(
                    index=id_cols,
                    columns=['variable', 'year'],
                    values='value',
                    aggfunc='first'  # In case of duplicates
                )
                
                print(f"Pivoted data: {self.df_pivot.shape[0]} scenarios, {len(self.df_pivot.columns)} variable-year combinations")
                print("Data is now in pivoted format with MultiIndex columns")
            else:
                # Check if columns have year patterns like "Variable (2030)"
                year_pattern_cols = [col for col in df.columns if '(' in str(col) and ')' in str(col)]
                if year_pattern_cols:
                    print("Detected year pattern in column names. Converting to MultiIndex...")
                    # Convert columns like "Variable (2030)" to MultiIndex
                    new_cols = []
                    for col in df.columns:
                        if '(' in str(col) and ')' in str(col):
                            parts = str(col).rsplit('(', 1)
                            var_name = parts[0].strip()
                            year_str = parts[1].rstrip(')').strip()
                            try:
                                year = int(year_str)
                                new_cols.append((var_name, year))
                            except ValueError:
                                new_cols.append((col, None))
                        else:
                            new_cols.append((col, None))
                    
                    df.columns = pd.MultiIndex.from_tuples(new_cols)
                    self.df_pivot = df
                    print("Converted to MultiIndex format")
                else:
                    raise ValueError(
                        "Data format not recognized. Expected one of:\n"
                        "1. Pivoted format with MultiIndex columns (variable, year)\n"
                        "2. Wide format with columns like 'Variable (2030)'\n"
                        "3. Long format with 'variable' and 'year' columns\n"
                        f"Found columns: {list(df.columns)}"
                    )
        
        return self.df_pivot
    
    def prepare_sector_variables(self, sector_assignment_path):
        """
        Prepare sector variable mapping.
        
        Parameters:
        -----------
        sector_assignment_path : str
            Path to CSV with Variable and Sector columns (required)
        """
        print("\n" + "="*60)
        print("Preparing Sector Variables")
        print("="*60)
        
        if not sector_assignment_path:
            raise ValueError("sector_assignment_path is required. Please provide --sector-assignment argument.")
        
        if not os.path.exists(sector_assignment_path):
            raise FileNotFoundError(f"Sector assignment file not found: {sector_assignment_path}")
        
        sector_df = pd.read_csv(sector_assignment_path)
        
        # Validate required columns
        if 'Variable' not in sector_df.columns or 'Sector' not in sector_df.columns:
            raise ValueError("Sector assignment file must have 'Variable' and 'Sector' columns")
        
        # Group by sector
        self.sector_vars = {}
        
        # Get available variables from df_pivot (handle both MultiIndex and regular columns)
        if isinstance(self.df_pivot.columns, pd.MultiIndex):
            available_vars_in_data = set(var for var, _ in self.df_pivot.columns)
        else:
            # For regular columns, extract variable names (remove year patterns)
            available_vars_in_data = set()
            for col in self.df_pivot.columns:
                if '(' in str(col) and ')' in str(col):
                    # Extract variable name from "Variable (2030)" format
                    var_name = str(col).rsplit('(', 1)[0].strip()
                    available_vars_in_data.add(var_name)
                else:
                    # If no year pattern, assume it's a variable name
                    available_vars_in_data.add(str(col))
        
        for sector in sector_df['Sector'].unique():
            vars_list = sector_df[sector_df['Sector'] == sector]['Variable'].tolist()
            # Filter to only variables that exist in df_pivot
            available_vars = [v for v in vars_list if v in available_vars_in_data]
            if available_vars:
                self.sector_vars[sector] = available_vars
                print(f"  {sector}: {len(available_vars)} variables")
            else:
                print(f"  Warning: No variables found for sector '{sector}' in the data")
        
        if not self.sector_vars:
            raise ValueError("No matching variables found between sector assignment file and data. Check variable names match.")
        
        return self.sector_vars
    
    def run_variable_selection(self, num_combinations=10, **kwargs):
        """
        Run variable selection to get top combinations.
        
        Parameters:
        -----------
        num_combinations : int, default=10
            Number of top combinations to return
        **kwargs : dict
            Additional arguments for select_variable_combinations
        """
        print("\n" + "="*60)
        print("Variable Selection")
        print("="*60)
        
        if self.sector_vars is None:
            raise ValueError("Sector variables must be prepared first. Call prepare_sector_variables() before run_variable_selection().")
        
        # Run variable selection
        results, sector_info = select_variable_combinations(
            self.df_pivot,
            self.sector_vars,
            num_samples=kwargs.get('num_samples', 1000),
            **{k: v for k, v in kwargs.items() if k != 'num_samples'}
        )
        
        # Get top N combinations
        top_combinations = results[:num_combinations]
        
        print(f"\nSelected top {len(top_combinations)} variable combinations")
        
        # Save summary
        summary_df = pd.DataFrame([
            {
                'combination': i+1,
                'n_variables': len(combo['variables']),
                'n_scenarios': combo['n_scenarios'],
                'max_correlation': combo['max_corr'],
                'variables': ', '.join(combo['variables'])
            }
            for i, combo in enumerate(top_combinations)
        ])
        summary_df.to_csv(
            os.path.join(self.output_base, 'variable_selection_summary.csv'),
            index=False
        )
        
        return top_combinations
    
    def run_single_combination(self, combo_idx, combo_data):
        """
        Run all analyses for a single variable combination.
        
        Parameters:
        -----------
        combo_idx : int
            Combination index (1-based)
        combo_data : dict
            Dictionary with 'variables', 'n_scenarios', 'max_corr'
        """
        print("\n" + "="*60)
        print(f"Running Analysis for Combination {combo_idx}")
        print("="*60)
        print(f"Variables ({len(combo_data['variables'])}): {combo_data['variables']}")
        print(f"Scenarios: {combo_data['n_scenarios']}")
        print(f"Max Correlation: {combo_data['max_corr']:.3f}")
        
        run_dir = os.path.join(self.runs_dir, f"combination_{combo_idx}")
        os.makedirs(run_dir, exist_ok=True)
        
        combo_vars = combo_data['variables']
        run_results = {
            'combination': combo_idx,
            'variables': combo_vars,
            'n_scenarios': combo_data['n_scenarios'],
            'max_correlation': combo_data['max_corr']
        }
        
        # 1. Filter and save data
        print("\n1. Filtering scenarios...")
        try:
            filtered_df = save_filtered_scenarios(
                self.df_pivot,
                combo_vars,
                years=[2030, 2050],
                threshold=0.8,
                save_path=os.path.join(run_dir, 'filtered_data.csv')
            )
            print(f"   Saved {len(filtered_df)} scenarios")
        except Exception as e:
            print(f"   Error filtering scenarios: {e}")
            return None
        
        # Create temp_bucket from Category if needed
        if 'temp_bucket' not in filtered_df.columns and 'Category' in filtered_df.columns:
            print("   Creating temp_bucket from Category column...")
            def categorize_temp_bucket(row):
                """Convert Category to temperature bucket."""
                if pd.isna(row.get('Category', None)):
                    return 'Unknown'
                cat = str(row['Category']).upper()
                if cat in ['C1', 'C2']:
                    return '1.5'
                elif cat in ['C3', 'C4']:
                    return '2.0'
                else:
                    return 'Above 2.0'
            
            filtered_df['temp_bucket'] = filtered_df.apply(categorize_temp_bucket, axis=1)
            print(f"   Created temp_bucket: {filtered_df['temp_bucket'].value_counts().to_dict()}")
        
        # 2. Prepare features and target
        print("\n2. Preparing features and target...")
        # Drop non-feature columns
        feature_cols = [col for col in filtered_df.columns 
                       if col not in ['model', 'scenario', 'region', 'Category', 'temp_bucket']]
        X = filtered_df[feature_cols]
        
        # Create target variable (binary: 1.5°C vs others)
        if 'temp_bucket' in filtered_df.columns:
            y = (filtered_df['temp_bucket'] == '1.5').astype(int)
        elif 'Category' in filtered_df.columns:
            y = (filtered_df['Category'].isin(['C1', 'C2'])).astype(int)
        else:
            print("   Warning: No temperature bucket column found. Skipping analysis.")
            return None
        
        # Scale and impute
        X_scaled, scaler = scale_and_impute(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        print(f"   Features: {X_scaled_df.shape[1]}, Samples: {X_scaled_df.shape[0]}")
        print(f"   Target distribution: {y.value_counts().to_dict()}")
        
        # 3. PRIM Analysis
        print("\n3. Running PRIM analysis...")
        try:
            prim_box, prim_lims, prim_stats = run_prim_analysis(
                X_scaled_df, y, threshold=0.5, visualize=False
            )
            
            # Save PRIM visualizations
            fig, ax = plt.subplots(figsize=(10, 6))
            prim_box.show_tradeoff(annotated=True)
            plt.savefig(os.path.join(self.visualizations_dir, f'prim_tradeoff_combo_{combo_idx}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            # Assess results
            prim_assessment = assess_prim_results(prim_stats)
            prim_assessment.to_csv(os.path.join(run_dir, 'prim_assessment.csv'), index=False)
            
            # Get restricted variables
            restricted_vars = get_restricted_variables(prim_box, X_scaled_df)
            prim_ranking = rank_prim_variables(prim_lims, restricted_vars)
            prim_ranking.to_csv(os.path.join(run_dir, 'prim_variable_ranking.csv'), index=False)
            
            run_results['prim'] = {
                'n_restricted_vars': len(restricted_vars),
                'restricted_vars': restricted_vars,
                'best_f1': prim_assessment['f1_score'].max() if len(prim_assessment) > 0 else 0
            }
            print(f"   PRIM: {len(restricted_vars)} restricted variables")
        except Exception as e:
            print(f"   Error in PRIM analysis: {e}")
            run_results['prim'] = {'error': str(e)}
        
        # 4. SHAP Analysis
        print("\n4. Running SHAP analysis...")
        try:
            shap_importance = run_shap_analysis(X_scaled_df, y, threshold=0.8)
            shap_importance.to_csv(os.path.join(run_dir, 'shap_importance.csv'), index=False)
            
            # Save SHAP plot
            from scripts.shap_analysis import plot_shap_summary
            from sklearn.model_selection import train_test_split
            from xgboost import XGBClassifier
            import shap
            
            X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.3, random_state=42)
            xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            xgb.fit(X_train, y_train)
            explainer = shap.Explainer(xgb)
            shap_values = explainer(X_scaled_df)
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_scaled_df, max_display=15, show=False)
            plt.savefig(os.path.join(self.visualizations_dir, f'shap_summary_combo_{combo_idx}.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            run_results['shap'] = {
                'top_10_vars': shap_importance.head(10)['Variable'].tolist(),
                'top_10_importance': shap_importance.head(10)['SHAP Importance'].tolist()
            }
            print(f"   SHAP: Top variable is {shap_importance.iloc[0]['Variable']}")
        except Exception as e:
            print(f"   Error in SHAP analysis: {e}")
            run_results['shap'] = {'error': str(e)}
        
        # 5. Random Forest Analysis
        print("\n5. Running Random Forest analysis...")
        try:
            rf_importances = run_random_forest(X_scaled_df, y, plot=False)
            rf_importances.to_csv(os.path.join(run_dir, 'rf_importances.csv'))
            
            # Save RF plot
            plt.figure(figsize=(10, 8))
            rf_importances.head(15).plot(kind='barh', title='Top Variables (Random Forest)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(self.visualizations_dir, f'rf_importances_combo_{combo_idx}.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            run_results['rf'] = {
                'top_10_vars': rf_importances.head(10).index.tolist(),
                'top_10_importance': rf_importances.head(10).values.tolist()
            }
            print(f"   RF: Top variable is {rf_importances.index[0]}")
        except Exception as e:
            print(f"   Error in RF analysis: {e}")
            run_results['rf'] = {'error': str(e)}
        
        # 6. PCA Analysis
        print("\n6. Running PCA analysis...")
        try:
            X_pca = apply_pca(X_scaled, n_components=5, plot=False)
            
            # Create dataframe with PCA components and target
            pca_df = pd.DataFrame(X_pca[:, :5], columns=[f'PC{i+1}' for i in range(5)])
            pca_df['temp_bucket'] = filtered_df['temp_bucket'].values if 'temp_bucket' in filtered_df.columns else y.values
            
            # Save 3D projection
            from mpl_toolkits.mplot3d import Axes3D
            from scipy.spatial import ConvexHull
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            
            temp_colors = {
                "1.5": "#D9420B",
                "2.0": "#035AA6",
                "Above 2.0": "#338474"
            }
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            for temp_label, hex_color in temp_colors.items():
                if temp_label in pca_df['temp_bucket'].values:
                    mask = pca_df['temp_bucket'] == temp_label
                    pts = X_pca[mask.values][:, 1:4] if X_pca.shape[1] >= 4 else X_pca[mask.values][:, :3]
                    if len(pts) > 0:
                        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=hex_color, alpha=0.2, label=f"{temp_label}°C")
            
            ax.set_title("3D PCA Projection with Temperature Buckets")
            ax.set_xlabel("PC2")
            ax.set_ylabel("PC3")
            ax.set_zlabel("PC4")
            ax.legend(title="Temperature Bucket", loc="upper left")
            ax.view_init(elev=10, azim=-100)
            plt.tight_layout()
            plt.savefig(os.path.join(self.visualizations_dir, f'pca_3d_combo_{combo_idx}.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            # Save explained variance plot
            from sklearn.decomposition import PCA
            pca = PCA(n_components=5)
            pca.fit(X_scaled)
            plt.figure(figsize=(8, 4))
            plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
            plt.xlabel("Number of Principal Components")
            plt.ylabel("Cumulative Explained Variance")
            plt.title("Explained Variance by PCA Components")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.visualizations_dir, f'pca_variance_combo_{combo_idx}.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            run_results['pca'] = {
                'explained_variance': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist()
            }
            print(f"   PCA: {np.sum(pca.explained_variance_ratio_):.2%} variance explained by 5 components")
        except Exception as e:
            print(f"   Error in PCA analysis: {e}")
            run_results['pca'] = {'error': str(e)}
        
        # 7. Correlation heatmap
        print("\n7. Generating correlation heatmap...")
        try:
            top_10_corr, _ = make_heat_map(self.df_pivot, combo_vars, years=[2030], plot=False)
            plt.figure(figsize=(12, 10))
            import seaborn as sns
            
            # Get correlation matrix
            selected_cols = [(v, 2030) for v in combo_vars if (v, 2030) in self.df_pivot.columns]
            if selected_cols:
                corr_data = self.df_pivot[selected_cols].corr()
                sns.heatmap(corr_data, cmap="coolwarm", annot=True, fmt=".2f", center=0)
                plt.title(f"Correlation Heatmap - Combination {combo_idx}")
                plt.tight_layout()
                plt.savefig(os.path.join(self.visualizations_dir, f'correlation_heatmap_combo_{combo_idx}.png'),
                           dpi=150, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"   Error generating heatmap: {e}")
        
        # Save run summary
        with open(os.path.join(run_dir, 'run_summary.json'), 'w') as f:
            json.dump(run_results, f, indent=2, default=str)
        
        return run_results
    
    def generate_report(self):
        """Generate comprehensive terminal report."""
        print("\n" + "="*80)
        print("ANALYSIS REPORT")
        print("="*80)
        print(f"\nOutput Directory: {self.output_base}")
        print(f"Total Combinations Analyzed: {len(self.results_summary)}")
        
        for result in self.results_summary:
            if result is None:
                continue
            print(f"\n{'='*80}")
            print(f"Combination {result['combination']}")
            print(f"{'='*80}")
            print(f"Variables: {len(result['variables'])}")
            print(f"Scenarios: {result['n_scenarios']}")
            print(f"Max Correlation: {result['max_correlation']:.3f}")
            
            if 'prim' in result and 'error' not in result['prim']:
                print(f"\nPRIM:")
                print(f"  Restricted Variables: {result['prim']['n_restricted_vars']}")
                print(f"  Best F1 Score: {result['prim']['best_f1']:.3f}")
            
            if 'shap' in result and 'error' not in result['shap']:
                print(f"\nSHAP Top Variable: {result['shap']['top_10_vars'][0]}")
            
            if 'rf' in result and 'error' not in result['rf']:
                print(f"\nRF Top Variable: {result['rf']['top_10_vars'][0]}")
            
            if 'pca' in result and 'error' not in result['pca']:
                cum_var = result['pca']['cumulative_variance'][-1]
                print(f"\nPCA: {cum_var:.2%} variance explained")
        
        print("\n" + "="*80)
        print("All results saved to:")
        print(f"  - Runs: {self.runs_dir}")
        print(f"  - Visualizations: {self.visualizations_dir}")
        print("="*80)
    
    def run_full_pipeline(self, num_combinations=10, sector_assignment_path=None, category_mapping_path=None, **kwargs):
        """
        Run the complete analysis pipeline.
        
        Parameters:
        -----------
        num_combinations : int, default=10
            Number of top variable combinations to analyze
        sector_assignment_path : str, required
            Path to sector assignment CSV file (required)
        category_mapping_path : str, optional
            Path to CSV file with model, scenario, and Category columns
        **kwargs : dict
            Additional arguments for variable selection
        """
        if not sector_assignment_path:
            raise ValueError("sector_assignment_path is required")
        # Load data
        self.load_data(category_mapping_path=category_mapping_path)
        
        # Prepare sectors
        self.prepare_sector_variables(sector_assignment_path)
        
        # Variable selection
        top_combinations = self.run_variable_selection(num_combinations=num_combinations, **kwargs)
        
        # Run analysis for each combination
        print("\n" + "="*80)
        print("RUNNING ANALYSIS FOR ALL COMBINATIONS")
        print("="*80)
        
        for idx, combo in enumerate(top_combinations, 1):
            result = self.run_single_combination(idx, combo)
            self.results_summary.append(result)
        
        # Generate report
        self.generate_report()
        
        # Save overall summary
        summary_path = os.path.join(self.output_base, 'analysis_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(self.results_summary, f, indent=2, default=str)
        
        print(f"\n✓ Analysis complete! Results saved to: {self.output_base}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run PRIM Analysis Pipeline')
    parser.add_argument('data_path', type=str, help='Path to input data CSV file')
    parser.add_argument('sector_assignment', type=str,
                       help='Path to sector assignment CSV file (required)')
    parser.add_argument('--category-mapping', type=str, default=None,
                       help='Path to CSV or Excel file (.xlsx) with model, scenario, and Category columns (optional, for joining Category from external source)')
    parser.add_argument('--output-folder', type=str, default=None,
                       help='Output folder name (will prompt if not provided)')
    parser.add_argument('--num-combinations', type=int, default=10,
                       help='Number of top combinations to analyze (default: 10)')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Number of samples for variable selection (default: 1000)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AnalysisPipeline(args.data_path, args.output_folder)
    
    # Run pipeline
    pipeline.run_full_pipeline(
        num_combinations=args.num_combinations,
        sector_assignment_path=args.sector_assignment,
        category_mapping_path=args.category_mapping,
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()
