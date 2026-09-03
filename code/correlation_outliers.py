"""Correlation and outlier analysis of networkTraffic.csv."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = "/home/john/cs741/Assignment2/assignment2/networkTraffic.csv"
RESULTS_DIR = "/home/john/cs741/Assignment2/results"
FIG_DIR = "/home/john/cs741/Assignment2/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({"font.size": 9, "font.family": "serif"})

df = pd.read_csv(DATA_PATH, na_values=["?"], low_memory=False)
target_col = "attack_cat"
numeric_cols = [c for c in df.columns if c not in ["proto", "state", "service", target_col, "id"]]

corr = df[numeric_cols].corr()
corr.to_csv(f"{RESULTS_DIR}/correlation_matrix.csv")

# find highly correlated pairs (|r| >= 0.9), excluding diagonal
pairs = []
cols = corr.columns
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        r = corr.iloc[i, j]
        if abs(r) >= 0.9:
            pairs.append((cols[i], cols[j], round(r, 4)))
pairs_df = pd.DataFrame(pairs, columns=["feature_1", "feature_2", "correlation"]).sort_values(
    "correlation", key=abs, ascending=False
)
pairs_df.to_csv(f"{RESULTS_DIR}/high_correlation_pairs.csv", index=False)
print("High correlation pairs (|r|>=0.9):\n", pairs_df.to_string())

# heatmap figure
fig, ax = plt.subplots(figsize=(7.0, 6.2))
sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, ax=ax, cbar_kws={"shrink": 0.7}, xticklabels=True, yticklabels=True)
ax.tick_params(axis="both", labelsize=5)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/correlation_heatmap.pdf")
plt.close()

# ---- outlier analysis via IQR ----
outlier_report = {}
for c in numeric_cols:
    col = df[c].dropna()
    q1, q3 = col.quantile(0.25), col.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        continue
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((col < lower) | (col > upper)).sum())
    pct_out = round(n_out / len(col) * 100, 3)
    outlier_report[c] = {"n_outliers": n_out, "pct_outliers": pct_out, "min": float(col.min()), "max": float(col.max())}

outlier_df = pd.DataFrame(outlier_report).T.sort_values("pct_outliers", ascending=False)
outlier_df.to_csv(f"{RESULTS_DIR}/outlier_summary.csv")
print("\nOutlier summary (top 15 by pct):\n", outlier_df.head(15))

# boxplots for a handful of representative high-outlier, high-scale features
sel = ["sbytes", "dbytes", "sload", "dload", "spkts", "dur"]
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2))
for ax, c in zip(axes.ravel(), sel):
    ax.boxplot(df[c].dropna(), vert=True, showfliers=True, flierprops={"markersize": 2})
    ax.set_title(c, fontsize=9)
    ax.set_yscale("symlog")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/outlier_boxplots.pdf")
plt.close()

# ---- class distribution figure ----
tvc = df[target_col].value_counts().sort_index()
map_df = pd.read_csv("/home/john/cs741/Assignment2/assignment2/attack_category_map.csv")
label_map = dict(zip(map_df["Mapping"], map_df["Attack Category Name"]))
labels = [label_map[i] for i in tvc.index]
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.bar(labels, tvc.values, color="#4C72B0")
ax.set_ylabel("Number of instances")
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/class_distribution.pdf")
plt.close()

# ---- feature scale figure (log scale of median abs value) ----
med = df[numeric_cols].abs().median().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(3.4, 2.8))
ax.barh(med.index[:15][::-1], med.values[:15][::-1], color="#55A868")
ax.set_xscale("log")
ax.set_xlabel("Median absolute value (log scale)")
ax.tick_params(axis="y", labelsize=6)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/feature_scale.pdf")
plt.close()

print("\nSaved correlation and outlier analysis outputs.")
