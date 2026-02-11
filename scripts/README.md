# Analysis Scripts

This folder contains separate scripts for different analysis methods used in the PRIM project, plus the main analysis pipeline.

**Standard Workflow:**
1. Pre-pivot your data: `python3 scripts/pivot_data.py data/your_data.csv --format parquet`
2. Run analysis: `python3 scripts/run_analysis.py data/your_data_pivoted.parquet data/categorized_variables.csv`

## Data Preprocessing: `pivot_data.py`

**Data Pivoting Script**

Converts long-format data (with `variable` and `year` columns) to wide-format pivoted data with MultiIndex columns. **Run this once** to preprocess your data for faster analysis runs.

### Usage

```bash
# Basic usage (saves as parquet by default)
python3 scripts/pivot_data.py data/combined_ar6_data.csv

# Specify output format
python3 scripts/pivot_data.py data/combined_ar6_data.csv --format parquet  # Recommended: compressed, fast
python3 scripts/pivot_data.py data/combined_ar6_data.csv --format csv     # Human-readable but large
python3 scripts/pivot_data.py data/combined_ar6_data.csv --format pickle  # Fastest but Python-only

# Specify custom output path
python3 scripts/pivot_data.py data/combined_ar6_data.csv --output data/combined_ar6_data_pivoted.parquet
```

### Output Formats

- **Parquet** (recommended): Compressed, fast to read/write, works with pandas
- **CSV**: Human-readable but very large files (not recommended for large datasets)
- **Pickle**: Fastest but Python-only, not portable

### Example

```bash
# Pivot the data once
python3 scripts/pivot_data.py data/combined_ar6_data.csv --format parquet

# This creates: data/combined_ar6_data_pivoted.parquet

# Then use the pivoted file in analysis (much faster!)
python3 scripts/run_analysis.py data/combined_ar6_data_pivoted.parquet data/categorized_variables.csv --output-folder my_analysis
```

**Note:** For large datasets (like your 216M row file), pivoting can take time. Run this once and reuse the pivoted file for all subsequent analyses.

---

## Main Pipeline: `run_analysis.py`

**Complete Analysis Pipeline**

The main script that orchestrates the entire analysis workflow from data loading to results generation.

### Quick Start

Run the complete analysis pipeline:

```bash
# Standard: Use pre-pivoted Parquet file
python3 scripts/run_analysis.py <pivoted_data_path> <sector_assignment> [options]
```

**Standard workflow:** Always use the pivoted Parquet file created by `pivot_data.py`.

### Example

```bash
# Standard workflow: Use pre-pivoted data (run pivot_data.py first)
python3 scripts/run_analysis.py data/combined_ar6_data_pivoted.parquet data/categorized_variables.csv --output-folder my_analysis --num-combinations 10
```

**Note:** 
- Paths are relative to where you run the command from. If you're in the PRIM directory, use `data/` not `../data/`
- **Standard approach:** Always use the pivoted Parquet file created by `pivot_data.py` for faster analysis

### Command Line Arguments

- `data_path` (required): Path to input data file (Parquet recommended, CSV/Pickle also supported)
- `sector_assignment` (required): Path to CSV file with Variable and Sector columns
- `--output-folder`: Name for output folder (will prompt if not provided)
- `--num-combinations`: Number of top variable combinations to analyze (default: 10)
- `--num-samples`: Number of samples for variable selection (default: 1000)

**Note:** Use the pivoted Parquet file created by `pivot_data.py` as the standard input format.

### Data Format Requirements

#### Input Data Format

Your input data file should be in one of these formats:

**Recommended:** Pre-pivot your data using `pivot_data.py` for faster analysis:
```bash
python3 scripts/pivot_data.py data/combined_ar6_data.csv --format parquet
python3 scripts/run_analysis.py data/combined_ar6_data_pivoted.parquet data/categorized_variables.csv ...
```

**Standard format (recommended):**

Use the pivoted Parquet file created by `pivot_data.py`:
- **Parquet** (`.parquet`): Compressed, fast, recommended standard format
- Already pivoted with MultiIndex columns `(variable_name, year)`
- Created by running: `python3 scripts/pivot_data.py data/your_data.csv --format parquet`

**Other supported formats:**

1. **Pivoted CSV/Pickle** (already pivoted):
   - **Pickle** (`.pkl`): Fastest but Python-only
   - **CSV** (`.csv`): Human-readable but large
   - Already pivoted with MultiIndex columns `(variable_name, year)`

2. **Long format** (automatically pivoted, slower - not recommended):
   - Columns: `model`, `scenario`, `variable`, `year`, `value` (and optionally `region`, `unit`, `Category`, `temp_bucket`)
   - The pipeline will automatically pivot this to wide format
   - **Note:** For large datasets, always use `pivot_data.py` first to create the standard Parquet file

3. **Year pattern in column names** (already pivoted):
   - Columns like `Variable Name (2030)`, `Variable Name (2050)`
   - Example: `Secondary Energy|Electricity (2030)`

**Required columns:**
- `model`: Model identifier
- `scenario`: Scenario identifier
- `variable`: Variable name (for long format)
- `year`: Year value (for long format)
- `value`: Variable value (for long format)
- `Category` or `temp_bucket`: Temperature category (C1/C2 for 1.5°C, C3/C4 for 2.0°C, etc.) - optional but recommended

#### Sector Assignment File (Required)

The sector assignment file groups variables into sectors (e.g., Electricity, Transport, Industry, AFOLU) for organized variable selection. This is required and helps ensure balanced combinations across different sectors.

**Format:** A CSV file with exactly two columns:
- `Variable`: Variable name (must match variable names in your data exactly)
- `Sector`: Sector name (e.g., "Electricity", "Transport", "Industry", "AFOLU", "Residential & Commercial")

**Example (`data/categorized_variables.csv`):**
```csv
Variable,Sector
Primary Energy|Biomass,AFOLU
Secondary Energy|Electricity,Electricity
Final Energy|Transportation,Transport
Final Energy|Industry,Industry
Final Energy|Residential and Commercial,Residential & Commercial
```

**How it's used:**
- The variable selection algorithm samples variables from each sector to create balanced combinations
- Ensures representation across different sectors (e.g., at least 2 variables from each sector)
- Required for the analysis pipeline to work properly

**Note:** Variable names must match exactly between your data and this file. For MultiIndex columns, match the variable part (first level); for "Variable (2030)" format, match the variable name before the parentheses.

#### Category Mapping File (Optional)

The Category mapping file provides temperature category classifications (C1, C2, C3, C4) for scenarios, as prescribed by independent bodies. This is optional but recommended if your data doesn't already include a `Category` column.

**Format:** A CSV or Excel file (`.xlsx`) with exactly three columns:
- `model`: Model identifier (must match model names in your data)
- `scenario`: Scenario identifier (must match scenario names in your data)
- `Category`: Temperature category (C1, C2, C3, C4, or other values)

**Example (`data/c_data.xlsx`):**
```csv
model,scenario,Category
AIM/CGE 2.0,ADVANCE_2020_1.5C-2100,C1
AIM/CGE 2.0,ADVANCE_2020_2C-2100,C3
...
```

**How it's used:**
- The pipeline joins this mapping with your data using `model` and `scenario` as keys
- Categories are automatically converted to temperature buckets (see `data/README.md` for full mapping):
  - C1, C2 → `1.5` (1.5°C scenarios)
  - C3, C4 → `2` (2.0°C scenarios)
  - C5, C6, C7, C8 → `above 2` (above 2.0°C scenarios)
  - `failed-vetting` → `failed-vetting` (scenarios that failed vetting)
  - `no-climate-assessment` → `no-climate-assessment` (scenarios without climate assessment)
- The `temp_bucket` column is then used for binary classification (1.5°C vs others)

**Note:** If your data already has a `Category` column, you don't need this file. The pipeline will automatically create `temp_bucket` from the existing `Category` column.

### Output Structure

The pipeline creates the following structure:

```
outputs/
└── [your-folder-name]/
    ├── variable_selection_summary.csv    # Summary of top combinations
    ├── analysis_summary.json             # Complete analysis summary
    ├── runs/                             # Individual combination results
    │   ├── combination_1/
    │   │   ├── filtered_data.csv
    │   │   ├── prim_assessment.csv
    │   │   ├── prim_variable_ranking.csv
    │   │   ├── shap_importance.csv
    │   │   ├── rf_importances.csv
    │   │   └── run_summary.json
    │   ├── combination_2/
    │   └── ...
    └── visualizations/                   # All plots
        ├── prim_tradeoff_combo_1.png
        ├── shap_summary_combo_1.png
        ├── rf_importances_combo_1.png
        ├── pca_3d_combo_1.png
        ├── pca_variance_combo_1.png
        ├── correlation_heatmap_combo_1.png
        └── ...
```

### Analysis Steps

For each of the top N variable combinations, the pipeline:

1. **Filters scenarios** based on completeness threshold (80%)
2. **Prepares features** by scaling and imputing missing values
3. **Runs PRIM analysis** to identify interpretable rules
4. **Runs SHAP analysis** to understand feature importance
5. **Runs Random Forest** for feature importance ranking
6. **Runs PCA** for dimensionality reduction visualization
7. **Generates correlation heatmaps** for variable relationships

### Terminal Report

The pipeline prints a comprehensive report showing:
- Variable selection summary
- Results for each combination
- Top variables from each analysis method
- Performance metrics

### Example Workflow

```bash
# Standard workflow: Pre-pivot data first (one-time)
python3 scripts/pivot_data.py data/my_data.csv --format parquet

# Then run analysis with pivoted data
# If your data doesn't have Category column, provide category mapping file:
python3 scripts/run_analysis.py data/my_data_pivoted.parquet data/categorized_variables.csv \
    --category-mapping data/c_data.xlsx \
    --output-folder experiment_1 \
    --num-combinations 10 \
    --num-samples 1000

# If your data already has Category column, omit --category-mapping:
python3 scripts/run_analysis.py data/my_data_pivoted.parquet data/categorized_variables.csv \
    --output-folder experiment_1 \
    --num-combinations 10
```

**Important:** 
- Always run the script from the PRIM project root directory
- Use `data/` for paths, not `../data/`
- **Standard approach:** Always pre-pivot your data using `pivot_data.py` and use the Parquet file for analysis

### Troubleshooting

#### Data Format Issues

If you get errors about data format:
- Ensure your data has MultiIndex columns or year patterns in column names
- Check that required columns (`model`, `scenario`, `Category`/`temp_bucket`) exist
- Verify variable names match exactly between data and sector assignment file
- Ensure sector assignment file has both 'Variable' and 'Sector' columns

#### Memory Issues

For large datasets:
- Reduce `--num-samples` to speed up variable selection
- Reduce `--num-combinations` to analyze fewer combinations
- Consider preprocessing data to reduce size before running

#### Missing Dependencies

**First time setup:**

1. Create a virtual environment (if you haven't already):
   ```bash
   python3 -m venv venv              # or any name you prefer (myenv, .venv, prim_env, etc.)
   source venv/bin/activate         # macOS/Linux (replace "venv" with your env name)
   ```

2. Install required packages from the requirements file:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

**Note:** You can name your virtual environment anything you like. Just remember to use that name when activating it!

**If virtual environment is already activated:**
```bash
pip install -r requirements.txt
```

**If you get a Parquet error (pyarrow not found):**
```bash
pip install pyarrow
```

Or install individually:
```bash
pip3 install pandas numpy matplotlib seaborn scikit-learn shap xgboost ema-workbench scipy pyarrow
```

**Note:** If you're using pyenv and `python` command is not found, use `python3` instead. You can also set up pyenv to use a specific Python version:
```bash
pyenv global 3.10.15  # or 3.13.0
```

### Pipeline Notes

- The pipeline automatically creates output directories
- All visualizations are saved (not displayed) for batch processing
- Results are saved in both CSV and JSON formats for flexibility
- The analysis can take significant time for large datasets (10+ combinations)

---

## Individual Analysis Modules

### 0. `variable_selection.py`
**Variable Selection**

Functions for selecting variables for PRIM analysis. The selection process considers coverage across scenarios and years, correlation between variables, sector-based organization, and completeness thresholds.

**Main Functions:**
- `select_variable_combinations()` - Main function for systematic variable selection
- `coverage_for_variable()` - Check variable coverage across years
- `variable_coverage_table()` - Generate coverage table for all variables
- `build_correlation_matrix()` - Build correlation matrix for a specific year
- `sample_variables()` - Sample initial variables from each sector
- `expand_variables()` - Expand variable combinations while maintaining correlation constraints
- `count_valid_scenarios()` - Count scenarios meeting completeness threshold
- `make_heat_map()` - Visualize correlation across selected variables
- `save_filtered_scenarios()` - Filter and save scenarios meeting completeness threshold

**Example Usage:**
```python
from scripts.variable_selection import select_variable_combinations

# Select variable combinations
results, sector_info = select_variable_combinations(
    df_pivot, 
    sector_vars, 
    years=[2030, 2050],
    num_samples=1000
)
```

---

### 1. `prim_analysis.py`
**PRIM (Patient Rule Induction Method) Analysis**

Performs PRIM analysis to identify interpretable rules (boxes) in variable space that distinguish temperature pathways.

**Main Functions:**
- `run_prim_analysis()` - Run PRIM analysis on scaled features
- `assess_prim_results()` - Assess PRIM box statistics and compute metrics
- `rank_prim_variables()` - Rank variables by importance in PRIM box
- `get_restricted_variables()` - Identify variables that PRIM actually restricted

**Example Usage:**
```python
from scripts.prim_analysis import run_prim_analysis, assess_prim_results

# Run PRIM analysis
box, lims, stats = run_prim_analysis(X_scaled, y, threshold=0.5)

# Assess results
results_df = assess_prim_results(stats)
print(results_df)
```

---

### 2. `shap_analysis.py`
**SHAP (SHapley Additive exPlanations) Analysis**

Performs SHAP analysis to understand feature importance and model interpretability using XGBoost.

**Main Functions:**
- `run_shap_analysis()` - Run SHAP analysis using XGBoost classifier
- `count_variables_to_reach_threshold()` - Count variables needed for threshold importance
- `plot_shap_summary()` - Plot SHAP summary plots

**Example Usage:**
```python
from scripts.shap_analysis import run_shap_analysis

# Run SHAP analysis
shap_importance = run_shap_analysis(X, y, threshold=0.8)
print(shap_importance.head(10))
```

---

### 3. `random_forest_analysis.py`
**Random Forest Analysis**

Performs Random Forest analysis for feature importance and cluster prediction.

**Main Functions:**
- `run_random_forest()` - Run Random Forest and extract feature importances
- `get_cluster_variable_mapping()` - Map variables to clusters using RF importance

**Example Usage:**
```python
from scripts.random_forest_analysis import run_random_forest

# Run Random Forest analysis
importances = run_random_forest(X_df, labels, n_estimators=200)
print(importances.head(10))
```

---

### 4. `pca_clustering_analysis.py`
**PCA and Clustering Analysis**

Performs Principal Component Analysis (PCA) and clustering to explore data structure and identify patterns.

**Main Functions:**
- `apply_pca()` - Apply PCA to scaled data
- `plot_3d_pca_projection()` - Create 3D projection of PCA results
- `plot_pca_pairplot()` - Create pairplot of PCA components
- `plot_cluster_boxplots()` - Plot boxplots for top variables by cluster
- `plot_combined_cluster_boxplots()` - Plot combined boxplots for multiple variables
- `scale_and_impute()` - Scale and impute missing values

**Example Usage:**
```python
from scripts.pca_clustering_analysis import apply_pca, plot_3d_pca_projection

# Apply PCA
X_pca = apply_pca(X_scaled, n_components=5)

# Plot 3D projection
plot_3d_pca_projection(df_pca, X_pca)
```

---

## Dependencies

All scripts require:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `shap` (for SHAP analysis)
- `xgboost` (for SHAP analysis)
- `ema_workbench` (for PRIM analysis)
- `scipy` (for PCA/clustering visualizations)

**Optional:**
- `ipyparallel` - For parallel evaluation in ema_workbench (not required for basic PRIM analysis). You may see a warning if it's not installed, but this won't affect functionality.

The scripts are self-contained and do not depend on `attempt_2/prim_utils.py`.

---

## Notes

- Each script is designed to be standalone and can be imported independently
- All scripts include example usage in their `main()` functions
- Visualizations are optional and can be disabled by setting `plot=False` or `visualize=False` where applicable
- Random seeds are set for reproducibility where applicable
