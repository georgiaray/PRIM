# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import defaultdict
import random

# Text processing
import nltk
from rapidfuzz import process, fuzz

# Machine learning & preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import shap
from xgboost import XGBClassifier

# Visualization
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

#PRIM
from ema_workbench.analysis.prim import Prim

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')



#functions
def coverage_for_variable(df_pivot, variable, years):
    '''
    This function checks the coverage of a variable across specified years in a DataFrame.
    It returns the number of complete rows for the specified variable across the years.
    :param df_pivot: DataFrame containing the data
    :param variable: The variable to check coverage for
    :param years: List of years to check coverage for 
    :return: Number of complete rows for the specified variable across the years
    '''
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
    '''
    This function generates a coverage table for all variables in the DataFrame across specified years.
    It returns a DataFrame with the variable names and their coverage counts.
    :param df_pivot: DataFrame containing the data
    :param years: List of years to check coverage for 
    :return: DataFrame with variable names and their coverage counts
    '''
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

def assign_sector(column_name, sector_keywords):
    col_lower = column_name.lower()
    
    # Check all sectors except Electricity first
    for sector in sector_keywords:
        if sector != "Electricity":
            for keyword in sector_keywords[sector]:
                if keyword.lower() in col_lower:
                    return sector

    # Then check Electricity last
    for keyword in sector_keywords["Electricity"]:
        if keyword.lower() in col_lower:
            return "Electricity"
    
    print(f"Unassigned column → '{column_name}' → sector: 'Other'")
    return "Other"

def build_correlation_matrix(df_pivot, year=2030):
    df_year = df_pivot.loc[:, df_pivot.columns.get_level_values(1) == year]
    corr_matrix = df_year.corr().abs()
    return corr_matrix

def sample_variables(sector_vars, min_vars=2, max_vars=5):
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
    combo = list(selected)

    for sector, remaining in sector_pool.items():
        current = [v for v in combo if v in sector_vars[sector]]
        while len(current) < max_vars and remaining:
            v = remaining.pop()
            test_combo = combo + [v]
            sub_corr = corr_matrix.loc[test_combo, test_combo].fillna(1)
            np.fill_diagonal(sub_corr.values, 0)
            if sub_corr.to_numpy().max() <= corr_threshold:
                combo.append(v)
                current.append(v)
    return combo

def is_below_correlation_threshold(combo, corr_matrix, threshold=0.9):
    sub_corr = corr_matrix.loc[combo, combo].fillna(1)
    np.fill_diagonal(sub_corr.values, 0)
    max_corr = sub_corr.to_numpy().max()
    return max_corr <= threshold, max_corr

def count_valid_scenarios(df_pivot, combo, years, threshold=0.8):
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
    Visualizes correlation across combo_vars for specified year(s).
    Keeps all columns (no dropping for completeness).
    Returns top 10 correlated pairs and the full scenario dataframe used.
    """
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
    Filters df_pivot to only include scenarios with >= threshold completeness
    for the given combo_vars and years. Optionally saves to CSV.

    Parameters:
        df_pivot (DataFrame): Pivoted scenario data with (variable, year) columns.
        combo_vars (List[str]): Variables to include.
        years (List[int]): Years to include.
        threshold (float): Required fraction of non-null values.
        save_path (str or None): Path to save filtered CSV. If None, only returns.

    Returns:
        DataFrame: Filtered scenarios with flattened column names.
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

def convert_to_2030_and_delta(df):
    # Keep descriptive columns unchanged
    keep_cols = ['model', 'scenario', 'Category', 'temp_bucket']
    new_df = df[keep_cols].copy() if all(col in df.columns for col in keep_cols) else df[['model', 'scenario']].copy()

    # Identify all 2030 and 2050 columns
    col_2030 = [col for col in df.columns if col.endswith('(2030)')]

    # Match 2030-2050 pairs by prefix
    col_pairs = []
    for col in col_2030:
        prefix = col.rsplit('(', 1)[0].strip()
        match_2050 = f"{prefix} (2050)"
        if match_2050 in df.columns:
            col_pairs.append((col, match_2050))

    # Create columns: Δ first, then 2030 value
    for col_30, col_50 in col_pairs:
        delta_name = 'Δ ' + col_30.replace(' (2030)', '').strip()
        new_df[delta_name] = df[col_50] - df[col_30]
        new_df[col_30] = df[col_30]

    return new_df

shortcut_dict = {'Secondary Energy': 'SecEng', 
 'Primary Energy': 'PriEng', 
 'Electricity': 'Elec',
 'Fossil': 'Foss', 
 'Transportation': 'Trans',
 'Passenger': 'Pass',
 'Final Energy': 'FinEng',
 'Industry': 'Ind',
 'Hydrogen': 'H2',
 'Demand': 'Dem',
 'Energy': 'Eng',
 'Residential and Commercial': 'ResCom',
 'Agriculture': 'Agri',
 'Non-Energy Crops and Livestock': 'NonEngCropStock',
 'Index': 'Idx',
 'Geothermal': 'Geo',
 'Carbon Sequestration': 'CSe',
 'Biomass': 'BioM',
 'Bioenergy': 'BioEng',
 'Residential': 'Res',
 'Land Cover': 'LandCov',
 'Cropland': 'Crop',
 'Irrigated': 'Irr',
 'Land Use': 'LandU',
 'Afforestation': 'Affor',
 '(2030)': ''
}

def data_exploration(df, display_corr = True):
    
    print("\nData Exploration:")
    print("-----------------------------------")

    print("\n Description of the DataFrame:")
    summary = df.describe()
    print("\nSummary statistics:\n", summary)

    print("------------------------------------")
    print("\n Missing Values:")

    missing = df.isnull().sum().sort_values(ascending=False)
    print("\nMissing values per column (top 10):\n", missing.head(10))

    print("------------------------------------")
    print("Dataset shape:", df.shape)

    if display_corr:
        #subset the columns that include '2030' and not '2050'
        vars_2030 = [col for col in df.columns if not col.startswith('Δ ') and df[col].dtype in [np.float64, np.int64]]

        df_2030 = df[vars_2030]

        plt.figure(figsize=(14, 12))
        corr = df_2030.drop(columns=['model', 'scenario'], errors='ignore').corr()
        sns.heatmap(corr, cmap="coolwarm", center=0, annot=True)
        plt.title("Correlation Heatmap, 2030 Variables")
        plt.tight_layout()
        plt.show()


def clean_text(s):
    if pd.isna(s):
        return s
    s = s.lower()
    s = re.sub(r'[\/\?\s]+', '_', s)  # Replace `/`, `?`, and whitespace with underscore
    s = re.sub(r'[^a-z0-9_\.\-]', '', s)  # Remove other special characters
    s = re.sub(r'_+', '_', s)  # Collapse multiple underscores
    s = s.strip('_')  # Remove leading/trailing underscores
    return s

def fuzzy_merge(df_main, df_ref, main_key='model_scenario', ref_key='model_scenario', threshold=90):

    # Step 1: Clean both keys
    df_main['match_key'] = df_main[main_key].apply(clean_text)
    df_ref['match_key'] = df_ref[ref_key].apply(clean_text)

    # Step 2: Perform fuzzy matching
    matches = []
    unmatched_main = []
    ref_values = df_ref['match_key'].dropna().unique().tolist()

    for value in df_main['match_key'].dropna().unique():
        match, score, _ = process.extractOne(value, ref_values, scorer=fuzz.token_sort_ratio)
        if score >= threshold:
            matches.append((value, match, score))
        else:
            unmatched_main.append(value)

    # Step 3: Create mapping DataFrame
    match_df = pd.DataFrame(matches, columns=['match_key', 'ref_match_key', 'match_score'])

    # Step 4: Merge matches into main
    df_merged = df_main.merge(match_df, on='match_key', how='left')

    # Merge matched keys from reference dataset
    df_final = df_merged.merge(
        df_ref,
        left_on='ref_match_key',
        right_on='match_key',
        how='left',
        suffixes=('', '_ref')
    )

    print(f"\nUnmatched entries in main dataset ({len(unmatched_main)}):")
    for item in unmatched_main:
        print(f"  - {item}")

    return df_final
    

sector_keywords = {
    "Electricity": ["electricity", "power", "grid", "Elec"],
    "Transport": ["transport", "vehicle", "freight", "passenger", "Trans", "Pass"],
    "Industry": ["industry", "cement", "steel", "manufacturing", "Ind"],
    "Buildings": ["building", "residential", "commercial", "Res", "ResCom"],
    "Agriculture": ["agriculture", "land", "livestock", "crop", "afolu", "Agri", "Crop", "LandU", "LandCov", "Affor"]
}

def create_sector_dfs(df): 
    #create separate dfs with just the columns relevant to each sector 
    column_sectors = []
    for column in df.columns: 
        sector = assign_sector(column, sector_keywords)
        column_sectors.append(sector)

    column_to_sector = dict(zip(df.columns, column_sectors))

    descriptive_cols = ['model', 'scenario', 'Category', 'temp_bucket']

    sector_column_map = defaultdict(list)
    for col, sector in column_to_sector.items():
        if sector and sector != 'descriptive' and col not in descriptive_cols:
            sector_column_map[sector].append(col)

    sector_dfs = {}
    for sector, cols in sector_column_map.items():
        sector_dfs[sector] = df[descriptive_cols + cols]

    return sector_dfs

def categorize_temp_bucket(row):
    if row['Category'] in ['C1', 'C2']:
        return '1.5'
    elif row['Category'] in ['C3', 'C4']:
        return '2.0'
    else:
        return 'Above 2.0'

def plot_temp_buckets(df):
    # Plot the distribution of temp buckets with the number on top
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x='temp_bucket', order=df['temp_bucket'].value_counts().index)

    # Annotate the counts on top of each bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')

    plt.title('Distribution of Temperature Buckets')
    plt.xlabel('Temperature Bucket')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def plot_sector_boxplots(df, sector_name, variables=None, temp_bucket_col='temp_bucket'):
    # Default to all non-descriptive columns if no variable list is provided
    if variables is None:
        descriptive_cols = ['model', 'scenario', 'Category', temp_bucket_col]
        variables = [col for col in df.columns if col not in descriptive_cols]

    # Melt the dataframe to long format for plotting
    sector_df = df.melt(id_vars=[temp_bucket_col], value_vars=variables,
                        var_name='Variable', value_name='Value')
    
    # drop any rows with NaN values
    sector_df = sector_df.dropna()

    # Create the boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=sector_df, x=temp_bucket_col, y='Value', hue='Variable', showfliers=False)

    # Set the title and labels
    plt.title(f'{sector_name} Sector Boxplot')
    plt.xlabel('Temperature Bucket')
    plt.ylabel('Value')
    plt.legend(title='Variable', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Display the plot
    plt.tight_layout()
    plt.show()

def logistic_regression(df):
    df['y'] = df['temp_bucket'].apply(lambda x: 1 if x in ['1.5', '2.0'] else 0)
    X = df.drop(columns=['model', 'scenario', 'Category', 'temp_bucket', 'y'])  # Drop y column for features
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    #fill in missing values with the mean of the column
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    X_train_scaled.fillna(X_train_scaled.mean(), inplace=True)
    X_test_scaled.fillna(X_test_scaled.mean(), inplace=True)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    return X_train_scaled, X_test_scaled, y_train, y_test, model

def log_reg_simple(X, y):
    model = LogisticRegression(max_iter=5000)
    model.fit(X, y)

    # Extract coefficients
    coefs = pd.DataFrame({
        "Variable": X.columns,
        "LogReg Coef": model.coef_[0]
    }).sort_values(by="LogReg Coef", key=abs, ascending=False)

    return model, coefs

def log_reg_report(y_test, y_pred):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

def visualize_log_reg(model, X, title='Logistic Regression Coefficients', top_bottom_n=None):
    coefficients = pd.DataFrame(model.coef_[0], index=X.columns, columns=['Coefficient'])
    
    if top_bottom_n is not None:
        # Get top n and bottom n coefficients
        top_n = coefficients.nlargest(top_bottom_n, 'Coefficient')
        bottom_n = coefficients.nsmallest(top_bottom_n, 'Coefficient')
        coefficients_to_plot = pd.concat([top_n, bottom_n]).sort_values(by='Coefficient', ascending=False)
    else:
        # Plot all coefficients
        coefficients_to_plot = coefficients.sort_values(by='Coefficient', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y=coefficients_to_plot.index, data=coefficients_to_plot)
    plt.title(title)
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    plt.show()

def balance_binary_classes(X, y, method="downsample"):
    X_majority = X[y == 1]
    X_minority = X[y == 0]

    if len(X_majority) < len(X_minority):
        # Invert the classes to always downsample the larger one
        X_majority, X_minority = X_minority, X_majority
        majority_label, minority_label = 0, 1
    else:
        majority_label, minority_label = 1, 0

    if method == "downsample":
        if len(X_majority) < len(X_minority):
            raise ValueError("Majority class must be larger than minority class for downsampling.")
        
        X_majority_downsampled = resample(
            X_majority,
            replace=False,
            n_samples=len(X_minority),
            random_state=42
        )
        X_balanced = pd.concat([X_majority_downsampled, X_minority])
        y_balanced = pd.Series(
            [majority_label] * len(X_minority) + [minority_label] * len(X_minority),
            index=X_balanced.index
        )
    else:
        raise ValueError("Only 'downsample' is supported.")

    return X_balanced, y_balanced

def get_x_and_Y(df, temp_target):
    df['y'] = df['temp_bucket'].apply(lambda x: 0 if x in temp_target else 1)

    X = df.drop(columns=['model', 'scenario', 'Category', 'temp_bucket', 'y'])
    y = df['y']

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    X_log = X.copy()
    for col in X.columns:
        if (X[col] > 0).all():  
            X_log[col] = np.log1p(X[col])
        else:
            X_log[col] = X[col] 

    X_log.fillna(X_log.mean(), inplace=True)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_log), columns=X_log.columns)

    return X_scaled, y

def prim_analysis(X_scaled, y, visualize=True):
    # Fit PRIM model
    p = Prim(X_scaled, y, threshold=0.5)
    box = p.find_box()

    if visualize:
        box.show_tradeoff(annotated=True)
        plt.show()
        box.show_pairs_scatter()
        plt.show()

    return box, box.box_lims[-1], p.stats

def assess(box_stats):
    results = []
    for i, stats in enumerate(box_stats):
        try:
            precision = stats['density']    # ← this is what EMA calls "mean" in summary
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

def rank_box_variables(lims, restricted_vars):
    importance = []

    for var, (low, high) in lims.items():
        if var in restricted_vars:
            range_width = np.inf if np.isinf(low) or np.isinf(high) else high - low
        else:
            low, high = -np.inf, np.inf
            range_width = np.inf
        importance.append((var, range_width))

    return pd.DataFrame(importance, columns=["Variable", "Box Range Width"]).sort_values("Box Range Width")

def multi_replace(colname, replacements):
    # Sort by length to avoid partial overlaps (e.g., "Energy" before "Bioenergy")
    sorted_keys = sorted(replacements, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(k) for k in sorted_keys))
    return pattern.sub(lambda m: replacements[m.group(0)], colname)

temp_colors = {
    "1.5": "#D9420B",
    "2.0": "#035AA6",
    "Above 2.0": "#338474"
}

def threed_projection(df_pca, X_pca):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for temp_label, hex_color in temp_colors.items():
        mask = df_pca['temp_bucket'] == temp_label
        pts = X_pca[mask.values][:, 1:4]  # Project on PC2, PC3, PC4

        # Scatter points
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=hex_color, alpha=0.2, label=f"{temp_label}°C")

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

def plt_sns(df_pca):
    sns.pairplot(df_pca, vars=df_pca.columns[:6], hue="Cluster", palette='viridis')
    plt.suptitle("Pairwise PCA Component Projections", y=1.02)
    plt.show()


def perform_rf(X_df, labels):
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_df, labels)

    importances = pd.Series(rf.feature_importances_, index=X_df.columns)
    top_importances = importances.sort_values(ascending=False)

    top_importances.plot(kind='barh', title='Top Variables Predicting Clusters')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    return top_importances

def jaccard_similarity(set1, set2):
    return len(set1 & set2) / len(set1 | set2)

def preprocess_data(df, exclude_columns=['model', 'scenario', 'Category', 'temp_bucket', 'y']):
    X = df.drop(columns=exclude_columns, errors='ignore')
    y = df['temp_bucket']
    return X, y

def train_decision_tree(X, y, max_leaf_nodes=6):
    y_binary = y.apply(lambda val: 'Above 1.5' if val in ['2.0', 'Above 2.0'] else val)

    le = LabelEncoder()
    print("Unique values in y:", y_binary.unique())
    print("Counts:\n", y_binary.value_counts())

    y_encoded = le.fit_transform(y_binary)
    tree = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=42)
    tree.fit(X, y_encoded)
    print("Class order:", le.classes_)
    return tree, le, y_encoded

def plot_decision_tree(tree, X_columns, class_names):
    plt.figure(figsize=(14, 8))
    plot_tree(tree, feature_names=X_columns, class_names=class_names, filled=True)
    plt.title("Decision Tree for Temperature Bucket Clustering")
    plt.tight_layout()
    plt.show()

def scale_and_impute(X):
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    return X_scaled, scaler

def apply_pca(X_scaled, n_components=15):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Explained Variance by PCA Components")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return X_pca


def plot_cluster_boxplots(df, labels, top_vars):
    for var in top_vars:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=labels, y=df[var])
        plt.title(f"{var} by Cluster")
        plt.xlabel("Cluster")
        plt.ylabel(var)
        plt.tight_layout()
        plt.show()

def count_variables_to_reach_threshold(shap_importance, threshold=0.8):
    sorted_importance = shap_importance["SHAP Importance"].sort_values(ascending=False).reset_index(drop=True)
    cumulative = sorted_importance.cumsum()
    total = sorted_importance.sum()
    num_vars = (cumulative / total < threshold).sum() + 1
    return num_vars

def shap_analysis(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    print("F1 Score:", f1_score(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    explainer = shap.Explainer(xgb)
    shap_values = explainer(X)

    # Summarize SHAP importance
    shap_importance = pd.DataFrame({
        "Variable": X.columns,
        "SHAP Importance": np.abs(shap_values.values).mean(axis=0)
    }).sort_values("SHAP Importance", ascending=False)

    top_k_80 = count_variables_to_reach_threshold(shap_importance, threshold=0.8)
    print(f"{top_k_80} variables account for 80% of total SHAP importance.")

    return shap_importance

def whole_shap_query(X, y, limits, restricted_vars): 
    _, coefs = log_reg_simple(X, y)
    shap_importance = shap_analysis(X, y)
    prim_df = rank_box_variables(limits, restricted_vars)

    # Merge all
    combined = prim_df.merge(coefs, on="Variable", how="outer")
    combined = combined.merge(shap_importance, on="Variable", how="outer")

    # Sort by any column you like
    combined_sorted = combined.sort_values("Box Range Width")  

    shap_75 = combined_sorted['SHAP Importance'].quantile(0.70)
    range_25 = combined_sorted['Box Range Width'].quantile(0.30)

    query = combined_sorted[
        (combined_sorted['SHAP Importance'] > shap_75) &
        (combined_sorted['Box Range Width'] < range_25)
    ]

    return query

def get_cluster_variable_mapping(df_clustered, cluster_to_bucket, top_n=10):

    results = []

    X = df_clustered.drop(columns=['model', 'scenario', 'Category', 'temp_bucket', 'cluster'])
    
    for cluster_id in sorted(df_clustered['cluster'].unique()):
        y = (df_clustered['cluster'] == cluster_id).astype(int)
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
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

    return pd.DataFrame(results)

def create_norm_df(combined_sorted, importance_cols):
    raw_df = combined_sorted[["Variable"] + importance_cols + ["Sector"]].set_index("Variable").copy()

    # Invert Box Range Width so lower = more important → higher score
    raw_df["Box Importance"] = 1 / (raw_df["Box Range Width"] + 1e-6)

    # Drop original Box Range Width to avoid confusion
    raw_df = raw_df.drop(columns=["Box Range Width"])

    # Reorder columns
    ordered_cols = ["Box Importance", "abs_logreg", "SHAP Importance", "Random Forest Importance"]

    # Normalize all scores to 0–1
    scaler = MinMaxScaler()
    normalized_values = scaler.fit_transform(raw_df[ordered_cols])

    # Create final normalized DataFrame
    normalized_df = pd.DataFrame(
        normalized_values,
        columns=[f"{col} (norm)" for col in ordered_cols],
        index=raw_df.index
    )

    #create an average score column
    normalized_df["Average Score"] = normalized_df.mean(axis=1)

    # Add sector back in
    normalized_df["Sector"] = raw_df["Sector"]

    

    return normalized_df

def get_restricted_variables(prim_box, X_scaled, tol=1e-6):
    """
    Prints the variables that PRIM actually restricted in the selected box.
    A variable is considered restricted if its lower or upper bound differs
    from the min/max in the input data (beyond a small tolerance).
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

cluster_to_bucket = {
    0: "1.5°C",
    1: "Above 1.5"
}

def plot_combined_cluster_boxplots(df, cluster_labels, variables, palette='Set2'):
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


