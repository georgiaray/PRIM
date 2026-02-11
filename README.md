# PRIM Project

Patient Rule Induction Method (PRIM) analysis for climate scenario data.

## Project Structure

```
PRIM/
├── requirements.txt      # Python package dependencies
├── venv/                 # Virtual environment (create with: python3 -m venv <name>)
├── scripts/              # Analysis modules
│   ├── run_analysis.py          # Main analysis pipeline script
│   ├── variable_selection.py    # Variable selection functions
│   ├── prim_analysis.py          # PRIM analysis
│   ├── shap_analysis.py          # SHAP analysis
│   ├── random_forest_analysis.py # Random Forest analysis
│   ├── pca_clustering_analysis.py # PCA and clustering
│   └── README.md                 # Detailed script documentation
├── data/                 # Data files (gitignored)
│   ├── ar6_data/         # AR6 dataset
│   ├── c_data.xlsx
│   ├── categorized_variables.csv
│   ├── combined_ar6_data.csv          # Original long-format data
│   └── combined_ar6_data_pivoted.parquet  # Standard pivoted format (created by pivot_data.py)
├── outputs/              # Analysis outputs
│   └── archive/          # Historical outputs
├── docs/                  # Documentation
└── README.md             # This file
```

## Quick Start

### First Time Setup

1. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv              # or any name you prefer (myenv, .venv, etc.)
   source venv/bin/activate          # macOS/Linux (use your env name)
   ```

2. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Preprocess your data (standard workflow - required):**
   ```bash
   python3 scripts/pivot_data.py data/combined_ar6_data.csv --format parquet
   ```
   This creates `data/combined_ar6_data_pivoted.parquet` - **this is the standard format** used for all analyses.

4. **You're ready to run analyses using the pivoted Parquet file!**

See the [Installation](#installation) section below for detailed instructions.

### Running the Complete Analysis Pipeline

The easiest way to run all analyses is using the main pipeline script:

```bash
# Standard workflow: Use pre-pivoted data (recommended)
python3 scripts/run_analysis.py data/combined_ar6_data_pivoted.parquet data/categorized_variables.csv --output-folder my_analysis
```

**Important:** 
- Always run the script from the PRIM project root directory
- Use `data/` for file paths, not `../data/`
- **Standard approach:** Pre-pivot your data once using `scripts/pivot_data.py`, then use the pivoted file for all analyses

This will:
1. Prompt for output folder name (if not provided)
2. Run variable selection to find top 10 combinations
3. Run PRIM, SHAP, RF, and PCA analysis for each combination
4. Save all results and visualizations
5. Generate a comprehensive terminal report

See `scripts/README.md` for detailed usage instructions.

### Individual Analysis Modules

You can also use individual modules:

```python
from scripts.variable_selection import select_variable_combinations

results, sector_info = select_variable_combinations(
    df_pivot, 
    sector_vars, 
    years=[2030, 2050],
    num_samples=1000
)
```

### PRIM Analysis

```python
from scripts.prim_analysis import run_prim_analysis, assess_prim_results

box, lims, stats = run_prim_analysis(X_scaled, y, threshold=0.5)
results_df = assess_prim_results(stats)
```

### SHAP Analysis

```python
from scripts.shap_analysis import run_shap_analysis

shap_importance = run_shap_analysis(X, y, threshold=0.8)
```

### Random Forest Analysis

```python
from scripts.random_forest_analysis import run_random_forest

importances = run_random_forest(X_df, labels, n_estimators=200)
```

### PCA and Clustering

```python
from scripts.pca_clustering_analysis import apply_pca, plot_3d_pca_projection

X_pca = apply_pca(X_scaled, n_components=5)
plot_3d_pca_projection(df_pca, X_pca)
```

## Installation

### Step 1: Clone or Navigate to the Repository

```bash
cd /path/to/PRIM
```

### Step 2: Create a Virtual Environment

Create a new virtual environment to isolate project dependencies. You can name it anything you like:

```bash
python3 -m venv venv
```

Or use a custom name:
```bash
python3 -m venv myenv
python3 -m venv .venv
python3 -m venv prim_env
```

This creates a folder (e.g., `venv`, `myenv`, `.venv`, `prim_env`) in your project directory.

### Step 3: Activate the Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate        # if named "venv"
source myenv/bin/activate       # if named "myenv"
source .venv/bin/activate       # if named ".venv"
source prim_env/bin/activate    # if named "prim_env"
```

**On Windows:**
```bash
venv\Scripts\activate        # if named "venv"
myenv\Scripts\activate       # if named "myenv"
.venv\Scripts\activate       # if named ".venv"
prim_env\Scripts\activate    # if named "prim_env"
```

You should see the environment name (e.g., `(venv)`, `(myenv)`, `(.venv)`) appear at the beginning of your terminal prompt, indicating the virtual environment is active.

### Step 4: Install Dependencies

Install all required packages from the requirements file:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `scikit-learn` - Machine learning
- `shap` - SHAP values for model interpretability
- `xgboost` - Gradient boosting
- `ema_workbench` - PRIM analysis
- `scipy` - Scientific computing
- `pyarrow` - Required for Parquet file support

### Step 5: Verify Installation

Verify that packages are installed correctly:

```bash
python3 -c "import pandas, numpy, sklearn, shap, xgboost, ema_workbench, pyarrow; print('All packages installed successfully!')"
```

**Note:** 
- You may see a warning about `ipyparallel` not being installed. This is optional and won't affect PRIM analysis functionality.
- If you get an error about `pyarrow`, install it: `pip install pyarrow` (required for Parquet file support)

### Step 6: Deactivate (When Done)

When you're finished working, deactivate the virtual environment:

```bash
deactivate
```

### Quick Setup Summary

```bash
# Navigate to project
cd /path/to/PRIM

# Create virtual environment (replace "venv" with your preferred name)
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux (replace "venv" with your env name)
# OR
venv\Scripts\activate     # Windows (replace "venv" with your env name)

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Pre-pivot your data (standard workflow - run once)
python3 scripts/pivot_data.py data/combined_ar6_data.csv --format parquet

# Verify installation
python3 -c "import pandas, numpy, sklearn, shap, xgboost, ema_workbench; print('Success!')"
```

**Note:** 
- You can name your virtual environment anything you like (e.g., `venv`, `myenv`, `.venv`, `prim_env`)
- Always activate your virtual environment before running the analysis scripts!
- **Standard workflow:** Always pre-pivot your data using `pivot_data.py` and use the Parquet file for analysis
- Replace `venv` in the activation commands with whatever name you chose

## Documentation

- See `scripts/README.md` for detailed documentation of all analysis modules and the main pipeline
- See `data/README.md` for information about data files
- See `outputs/README.md` for information about output structure

## Notes

- Large data files are gitignored to keep repository size manageable
- All analysis scripts are modular and can be imported independently
- Historical outputs are archived in `outputs/archive/`
