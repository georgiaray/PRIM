"""
Data Pivoting Script

This script converts long-format data (with variable and year columns) 
to wide-format pivoted data with MultiIndex columns.
Run this once to preprocess your data for faster analysis runs.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path


def pivot_data(input_path, output_path=None, format='csv'):
    """
    Pivot long-format data to wide format with MultiIndex columns.
    
    Parameters:
    -----------
    input_path : str
        Path to input CSV file in long format
    output_path : str, optional
        Path to save pivoted data. If None, appends '_pivoted' to input filename
    format : str, default='csv'
        Output format: 'csv', 'parquet', or 'pickle'
        - 'csv': Human-readable but large files
        - 'parquet': Compressed, fast, recommended for large datasets
        - 'pickle': Python-native, fastest but not portable
        
    Returns:
    --------
    pd.DataFrame
        Pivoted DataFrame with MultiIndex columns
    """
    print("="*60)
    print("Data Pivoting Script")
    print("="*60)
    print(f"\nLoading data from: {input_path}")
    
    # Load data (support CSV and Parquet)
    file_ext = os.path.splitext(input_path)[1].lower()
    if file_ext == '.parquet':
        try:
            df = pd.read_parquet(input_path)
            print(f"Loaded data from Parquet file")
        except ImportError:
            raise ImportError(
                "pyarrow is required for Parquet support. Install it with:\n"
                "  pip install pyarrow"
            )
    else:
        df = pd.read_csv(input_path)
        print(f"Loaded data from CSV file")
    
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Normalize column names to lowercase for matching (preserve year columns as-is)
    column_mapping = {}
    for col in df.columns:
        col_str = str(col)
        col_lower = col_str.lower()
        # Normalize standard columns to lowercase, but preserve year columns (4-digit numbers)
        if col_lower in ['model', 'scenario', 'region', 'variable', 'unit', 'category']:
            column_mapping[col] = col_lower
        # Keep year columns (4-digit numbers) and other columns as-is
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
        print(f"Normalized column names: {column_mapping}")
        print(f"  Category normalized: {'Category' in column_mapping or 'category' in column_mapping.values()}")
    
    # Check if data is already in long format (has 'variable' and 'year' columns)
    has_long_format = 'variable' in df.columns and 'year' in df.columns and 'value' in df.columns
    
    # Check if data is in semi-wide format (has 'variable' column and year columns)
    year_cols = [col for col in df.columns if str(col).isdigit() and len(str(col)) == 4]
    has_semi_wide_format = 'variable' in df.columns and len(year_cols) > 0
    
    if not has_long_format and not has_semi_wide_format:
        # Check if it's already fully pivoted (MultiIndex columns or year pattern)
        if isinstance(df.columns, pd.MultiIndex):
            print("\nData is already in pivoted format with MultiIndex columns!")
            print("No pivoting needed. You can use this file directly for analysis.")
            return df
        
        year_pattern_cols = [col for col in df.columns if '(' in str(col) and ')' in str(col)]
        if year_pattern_cols:
            print("\nData appears to be in pivoted format with year patterns in column names!")
            print("No pivoting needed. You can use this file directly for analysis.")
            return df
        
        # Neither format - error
        raise ValueError(
            f"Data format not recognized. Expected one of:\n"
            f"  1. Long format: columns ['model', 'scenario', 'variable', 'year', 'value']\n"
            f"  2. Semi-wide format: columns ['Model', 'Scenario', 'Variable', '1995', '1996', ...]\n"
            f"  3. Already pivoted: MultiIndex columns or 'Variable (Year)' pattern\n"
            f"\nFound columns: {list(df.columns)}"
        )
    
    # Convert semi-wide format to long format if needed
    if has_semi_wide_format:
        print(f"\nDetected semi-wide format (years as columns). Converting to long format...")
        print(f"  Found {len(year_cols)} year columns: {year_cols[:5]}{'...' if len(year_cols) > 5 else ''}")
        
        # Identify ID columns (everything except year columns)
        id_cols = [col for col in df.columns if col not in year_cols]
        print(f"  ID columns: {id_cols}")
        print(f"  Category in ID columns: {'category' in [c.lower() for c in id_cols]}")
        
        # Melt year columns into long format
        df = df.melt(
            id_vars=id_cols,
            value_vars=year_cols,
            var_name='year',
            value_name='value'
        )
        
        # Convert year to integer
        df['year'] = df['year'].astype(int)
        
        print(f"  Converted to long format: {len(df):,} rows")
        print(f"  Columns after melt: {list(df.columns)}")
        print(f"  Category in columns after melt: {'category' in [c.lower() for c in df.columns]}")
    
    # Now validate we have long format
    required_cols = ['model', 'scenario', 'variable', 'year', 'value']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"After conversion, missing required columns: {missing_cols}")
    
    # Determine index columns (check both lowercase and original case)
    id_cols = ['model', 'scenario']
    optional_id_cols = ['region', 'category', 'temp_bucket']  # Use lowercase since we normalized
    df_cols_lower = {str(col).lower(): col for col in df.columns}
    
    for col_lower in optional_id_cols:
        if col_lower in df_cols_lower:
            # Use the actual column name (might be 'Category' or 'category')
            actual_col = df_cols_lower[col_lower]
            id_cols.append(actual_col)
    
    print(f"\nUsing index columns: {id_cols}")
    print(f"Pivoting on: variable, year")
    print(f"  Category in columns: {'category' in df_cols_lower}")
    print(f"  temp_bucket in columns: {'temp_bucket' in df_cols_lower}")
    
    # Handle temp_bucket: use existing if present, otherwise create from Category
    category_col = df_cols_lower.get('category')
    temp_bucket_col = df_cols_lower.get('temp_bucket')
    
    if temp_bucket_col:
        # temp_bucket already exists - use it as-is
        print(f"\ntemp_bucket column already exists in data. Using existing values.")
        print(f"  temp_bucket distribution: {df[temp_bucket_col].value_counts().to_dict()}")
        # Ensure temp_bucket is in id_cols (should already be there from earlier)
        if temp_bucket_col not in id_cols:
            id_cols.append(temp_bucket_col)
        # Remove Category from id_cols if both exist (prefer temp_bucket)
        if category_col and category_col in id_cols:
            id_cols.remove(category_col)
            print(f"  Removed Category from index (using temp_bucket instead)")
    elif category_col:
        # temp_bucket doesn't exist - create it from Category
        print(f"\nCreating temp_bucket from Category (column: '{category_col}')...")
        # Mapping from Category to temp_bucket (as defined in data/README.md)
        category_to_temp_mapping = {
            'C1': '1.5', 
            'C2': '1.5', 
            'C3': '2', 
            'C4': '2', 
            'C5': 'above 2', 
            'C6': 'above 2', 
            'C7': 'above 2', 
            'C8': 'above 2', 
            'failed-vetting': 'failed-vetting', 
            'no-climate-assessment': 'no-climate-assessment'
        }
        
        df['temp_bucket'] = df[category_col].apply(
            lambda x: category_to_temp_mapping.get(str(x).strip(), x) if pd.notna(x) else None
        )
        print(f"  Created temp_bucket: {df['temp_bucket'].value_counts().to_dict()}")
        
        # Add temp_bucket to id_cols and remove Category (using actual column name)
        if category_col in id_cols:
            id_cols.remove(category_col)
        if 'temp_bucket' not in id_cols:
            id_cols.append('temp_bucket')
        
        print(f"  Updated index columns: {id_cols}")
    else:
        print("\n  Note: Neither Category nor temp_bucket found in data.")
        print("  temp_bucket will not be included in the pivoted index.")
    
    # Pivot the data
    print("\nPivoting data (this may take a while for large datasets)...")
    df_pivot = df.pivot_table(
        index=id_cols,
        columns=['variable', 'year'],
        values='value',
        aggfunc='first'  # In case of duplicates
    )
    
    print(f"\n✓ Pivoted successfully!")
    print(f"  Scenarios: {df_pivot.shape[0]:,}")
    print(f"  Variable-year combinations: {len(df_pivot.columns):,}")
    print(f"  Memory usage: {df_pivot.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
    
    # Determine output path
    if output_path is None:
        input_path_obj = Path(input_path)
        if format == 'csv':
            output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_pivoted.csv")
        elif format == 'parquet':
            output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_pivoted.parquet")
        elif format == 'pickle':
            output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_pivoted.pkl")
        else:
            raise ValueError(f"Unknown format: {format}")
    
    # Save pivoted data
    print(f"\nSaving pivoted data to: {output_path}")
    
    if format == 'csv':
        df_pivot.to_csv(output_path)
        print("✓ Saved as CSV")
    elif format == 'parquet':
        try:
            df_pivot.to_parquet(output_path, compression='snappy')
            print("✓ Saved as Parquet (compressed)")
        except ImportError:
            raise ImportError(
                "pyarrow is required for Parquet support. Install it with:\n"
                "  pip install pyarrow\n"
                "Or use --format csv or --format pickle instead."
            )
    elif format == 'pickle':
        df_pivot.to_pickle(output_path)
        print("✓ Saved as Pickle")
    
    print(f"\nFile size: {os.path.getsize(output_path) / 1024**3:.2f} GB")
    print(f"\n✓ Pivoted data saved successfully!")
    print(f"\nStandard workflow - use this file for all analyses:")
    print(f"  python3 scripts/run_analysis.py {output_path} data/categorized_variables.csv --output-folder my_analysis")
    
    return df_pivot


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Pivot long-format data to wide format for PRIM analysis'
    )
    parser.add_argument('input_path', type=str,
                       help='Path to input CSV or Parquet file in long format')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file path (default: adds _pivoted to input filename)')
    parser.add_argument('--format', '-f', type=str, default='parquet',
                       choices=['csv', 'parquet', 'pickle'],
                       help='Output format: csv (readable), parquet (compressed, recommended), pickle (fastest)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input file not found: {args.input_path}")
    
    # Pivot and save
    pivot_data(args.input_path, args.output, args.format)


if __name__ == "__main__":
    main()
