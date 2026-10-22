"""
Customer Churn Intelligence Platform
Phase 1: Dataset Inspection and Data Quality Analysis Script

This script inspects the Telco Customer Churn dataset, analyzes its structure,
checks for missing/invalid values, examines the target distribution,
and identifies data-quality issues without modifying the original raw CSV.
"""

from pathlib import Path
import warnings
import pandas as pd

# Suppress pandas migration notices for select_dtypes across versions
warnings.filterwarnings("ignore", category=UserWarning)


def get_dataset_path() -> Path:
    """Locate the dataset file whether run from root or src directory."""
    candidates = [
        Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv"),
        Path("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"),
        Path("WA_Fn-UseC_-Telco-Customer-Churn.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find WA_Fn-UseC_-Telco-Customer-Churn.csv")


def inspect_dataset():
    csv_path = get_dataset_path()
    print("=" * 80)
    print("CUSTOMER CHURN INTELLIGENCE PLATFORM - PHASE 1 INSPECTION")
    print("=" * 80)
    print(f"[OK] Dataset located successfully at: {csv_path.resolve()}\n")

    # 1. Load dataset (read-only)
    df = pd.read_csv(csv_path)

    # 2. Basic Dimensions
    rows, cols = df.shape
    print("--- 1. DATASET DIMENSIONS ---")
    print(f"Total Rows (Customers): {rows:,}")
    print(f"Total Columns (Features): {cols:,}\n")

    # 3. Column Names and Data Types
    print("--- 2. COLUMN NAMES & DATA TYPES ---")
    schema_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": [str(t) for t in df.dtypes],
        "Non-Null Count": df.notnull().sum().values,
        "Sample Value": [df[c].iloc[0] for c in df.columns]
    })
    print(schema_df.to_string(index=False))
    print()

    # 4. First 10 Rows
    print("--- 3. FIRST 10 ROWS PREVIEW ---")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(df.head(10).to_string())
    print("\n")

    # 5. Missing Values & Duplicate Records
    print("--- 4. MISSING VALUES & DUPLICATES CHECK ---")
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate Rows: {duplicate_count}")

    id_unique = df["customerID"].nunique()
    print(f"Unique Customer IDs: {id_unique:,} out of {rows:,} rows")
    if id_unique == rows:
        print("[OK] All customerID values are 100% unique (Primary Key confirmed).")
    else:
        print(f"[WARN] Warning: Found {rows - id_unique} duplicate customer IDs.")

    explicit_missing = df.isnull().sum()
    explicit_missing_cols = explicit_missing[explicit_missing > 0]
    if len(explicit_missing_cols) == 0:
        print("[INFO] Standard df.isnull() detected 0 explicit missing values (Note: inspect whitespace strings).")
    else:
        print(explicit_missing_cols)
    print()

    # 6. Deep Dive into TotalCharges
    print("--- 5. INVESTIGATION OF 'TotalCharges' COLUMN ---")
    print(f"Declared Data Type: {df['TotalCharges'].dtype}")
    
    # Check for whitespace/empty strings
    whitespace_mask = df["TotalCharges"].str.strip() == ""
    whitespace_count = whitespace_mask.sum()
    print(f"Blank / Whitespace strings detected: {whitespace_count}")

    if whitespace_count > 0:
        problem_rows = df[whitespace_mask][["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]]
        print("\nDetailed view of rows with blank TotalCharges:")
        print(problem_rows.to_string(index=False))
        
        # Test numeric conversion with coerce
        converted = pd.to_numeric(df["TotalCharges"], errors="coerce")
        nan_count = converted.isna().sum()
        print(f"\nAfter converting to numeric (pd.to_numeric with errors='coerce'):")
        print(f"Resulting NaN count: {nan_count} (matches blank count: {nan_count == whitespace_count})")
        print("Root cause observation: Notice that all these customers have tenure = 0 (newly joined customers).")
    print()

    # 7. Target Column Analysis (Churn)
    print("--- 6. TARGET COLUMN ('Churn') ANALYSIS ---")
    churn_counts = df["Churn"].value_counts()
    churn_pct = df["Churn"].value_counts(normalize=True) * 100
    target_summary = pd.DataFrame({
        "Customer Count": churn_counts,
        "Percentage (%)": churn_pct.round(2)
    })
    print(target_summary.to_string())
    imbalance_ratio = churn_counts["No"] / churn_counts["Yes"]
    print(f"\nClass Ratio (Retained to Churned): {imbalance_ratio:.2f}:1")
    print(f"Churn Rate: {churn_pct['Yes']:.2f}% (Moderate class imbalance - stratified splitting recommended).")
    print()

    # 8. Unique Values and Categorical Inconsistencies Check
    print("--- 7. CATEGORICAL VALUES & QUALITY CHECK ---")
    categorical_cols = [c for c in df.columns if c not in ["customerID", "TotalCharges", "tenure", "MonthlyCharges", "SeniorCitizen"]]

    for col in categorical_cols:
        vals = df[col].unique().tolist()
        print(f"- {col} ({len(vals)} unique): {vals}")

    print("\n--- 8. NUMERIC RANGES & ANOMALIES CHECK ---")
    print(f"SeniorCitizen unique values: {df['SeniorCitizen'].unique().tolist()} (Encoded as 0/1 integer)")
    print(f"Tenure range: min = {df['tenure'].min()}, max = {df['tenure'].max()} months")
    print(f"MonthlyCharges range: min = ${df['MonthlyCharges'].min():.2f}, max = ${df['MonthlyCharges'].max():.2f}")
    
    # Check numeric conversion for TotalCharges summary stats
    total_charges_numeric = pd.to_numeric(df["TotalCharges"], errors="coerce")
    print(f"TotalCharges (valid numeric only): min = ${total_charges_numeric.min():.2f}, max = ${total_charges_numeric.max():.2f}, mean = ${total_charges_numeric.mean():.2f}")

    # Tenure = 0 count
    zero_tenure_count = (df["tenure"] == 0).sum()
    print(f"Customers with tenure = 0: {zero_tenure_count} (Matches exactly with blank TotalCharges count: {zero_tenure_count == whitespace_count})")

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE. RAW CSV REMAINS 100% UNCHANGED.")
    print("=" * 80)


if __name__ == "__main__":
    inspect_dataset()
