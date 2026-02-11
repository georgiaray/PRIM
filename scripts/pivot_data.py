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
    
    # Load data
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Validate required columns
    required_cols = ['model', 'scenario', 'variable', 'year', 'value']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Determine index columns
    id_cols = ['model', 'scenario']
    optional_id_cols = ['region', 'Category', 'temp_bucket']
    for col in optional_id_cols:
        if col in df.columns:
            id_cols.append(col)
    
    print(f"\nUsing index columns: {id_cols}")
    print(f"Pivoting on: variable, year")
    
    # Create temp_bucket from Category before pivoting if Category exists
    if 'Category' in df.columns and 'temp_bucket' not in df.columns:
        print("\nCreating temp_bucket from Category...")
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
                       help='Path to input CSV file in long format')
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
