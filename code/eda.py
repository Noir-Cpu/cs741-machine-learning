"""Exploratory data analysis of networkTraffic.csv for CS741 Assignment 2."""
import pandas as pd
import numpy as np
import json
import os

DATA_PATH = "/home/john/cs741/Assignment2/assignment2/networkTraffic.csv"
RESULTS_DIR = "/home/john/cs741/Assignment2/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, na_values=["?"], low_memory=False)
print("Shape:", df.shape)
print(df.dtypes)

report = {}
report["n_rows"] = int(df.shape[0])
report["n_cols"] = int(df.shape[1])
report["columns"] = list(df.columns)

# ---- Missing values ----
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(3)
missing_table = pd.DataFrame({"n_missing": missing, "pct_missing": missing_pct})
missing_table.to_csv(f"{RESULTS_DIR}/missing_values.csv")
print("\nMissing values:\n", missing_table)

# rows with too many missing features
row_missing_counts = df.isna().sum(axis=1)
report["row_missing_count_distribution"] = row_missing_counts.value_counts().sort_index().to_dict()
report["rows_missing_ge_3"] = int((row_missing_counts >= 3).sum())
report["rows_missing_ge_5"] = int((row_missing_counts >= 5).sum())

# ---- dtypes / attempted numeric conversion for object columns ----
obj_cols = df.select_dtypes(include="object").columns.tolist()
print("\nObject columns:", obj_cols)
report["object_columns"] = obj_cols

# ---- cardinality ----
card = {c: int(df[c].nunique(dropna=True)) for c in df.columns}
card_table = pd.Series(card).sort_values()
card_table.to_csv(f"{RESULTS_DIR}/cardinality.csv")
print("\nCardinality (lowest 15):\n", card_table.head(15))

# ---- nominal feature value counts ----
for c in ["proto", "state", "service"]:
    vc = df[c].value_counts(dropna=False)
    vc.to_csv(f"{RESULTS_DIR}/valuecounts_{c}.csv")
    print(f"\n{c} value counts (top 10):\n", vc.head(10))

# ---- target distribution ----
target_col = df.columns[-1]
report["target_col"] = target_col
tvc = df[target_col].value_counts(dropna=False)
tvc.to_csv(f"{RESULTS_DIR}/target_distribution.csv")
print("\nTarget distribution:\n", tvc)
report["target_missing"] = int(df[target_col].isna().sum())

# ---- id uniqueness ----
report["id_unique"] = bool(df["id"].nunique() == len(df))
report["id_dtype"] = str(df["id"].dtype)

# ---- numeric summary (describe) for likely-numeric columns ----
numeric_candidates = [c for c in df.columns if c not in ["proto", "state", "service", target_col]]
df_num = df[numeric_candidates].apply(pd.to_numeric, errors="coerce")
desc = df_num.describe().T
desc["n_non_numeric_coerced_nan"] = (df[numeric_candidates].isna().sum() - df_num.isna().sum()).abs()
desc.to_csv(f"{RESULTS_DIR}/numeric_describe.csv")
print("\nNumeric describe (head):\n", desc.head(20))

# check which columns had non-numeric junk values (like A10 issue)
for c in numeric_candidates:
    orig_non_na = df[c].notna().sum()
    coerced_non_na = df_num[c].notna().sum()
    if orig_non_na != coerced_non_na:
        bad_vals = df.loc[df[c].notna() & df_num[c].isna(), c].unique()
        print(f"Column {c} has {orig_non_na - coerced_non_na} non-numeric entries, examples: {bad_vals[:10]}")
        report.setdefault("non_numeric_junk", {})[c] = {
            "count": int(orig_non_na - coerced_non_na),
            "examples": [str(x) for x in bad_vals[:10]],
        }

# ---- negative values check for columns expected non-negative ----
neg_report = {}
for c in numeric_candidates:
    col = df_num[c]
    n_neg = int((col < 0).sum())
    if n_neg > 0:
        neg_report[c] = n_neg
report["negative_value_columns"] = neg_report
print("\nColumns with negative values:", neg_report)

# ---- scale / magnitude comparison (median of abs values) ----
scale = df_num.abs().median().sort_values(ascending=False)
scale.to_csv(f"{RESULTS_DIR}/feature_scale_median.csv")
print("\nFeature scale (median abs value), top 10:\n", scale.head(10))

# ---- duplicated rows ----
report["n_duplicate_rows"] = int(df.duplicated().sum())

with open(f"{RESULTS_DIR}/eda_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

print("\nSaved EDA outputs to", RESULTS_DIR)
