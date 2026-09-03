"""Generate result figures (confusion matrices, feature importance, model
comparison) for the report, from the CSV/JSON outputs of run_knn.py and
run_tree.py."""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "/home/john/cs741/Assignment2/results"
FIG_DIR = "/home/john/cs741/Assignment2/figures"
plt.rcParams.update({"font.size": 9, "font.family": "serif"})

map_df = pd.read_csv("/home/john/cs741/Assignment2/assignment2/attack_category_map.csv")
label_map = dict(zip(map_df["Mapping"], map_df["Attack Category Name"]))
labels = [label_map[i] for i in range(10)]


def plot_confusion(name):
    path = f"{RESULTS_DIR}/{name}_confusion_matrix.csv"
    if not os.path.exists(path):
        print("skip", path)
        return
    cm = np.loadtxt(path, delimiter=",")
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    sns.heatmap(cm_norm, annot=False, cmap="Blues", vmin=0, vmax=1, ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_xticks(np.arange(10) + 0.5)
    ax.set_yticks(np.arange(10) + 0.5)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, rotation=0, fontsize=6)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{name}_confusion_matrix.pdf")
    plt.close()
    print("wrote", f"{FIG_DIR}/{name}_confusion_matrix.pdf")


def plot_feature_importance():
    path = f"{RESULTS_DIR}/tree_feature_importances.csv"
    if not os.path.exists(path):
        print("skip", path)
        return
    df = pd.read_csv(path).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.barh(df["feature"], df["importance"], color="#4C72B0")
    ax.set_xlabel("Gini/entropy importance")
    ax.tick_params(axis="y", labelsize=6)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/tree_feature_importance.pdf")
    plt.close()
    print("wrote", f"{FIG_DIR}/tree_feature_importance.pdf")


def plot_knn_grid():
    path = f"{RESULTS_DIR}/knn_grid_search_results.csv"
    if not os.path.exists(path):
        print("skip", path)
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for (weights, metric), g in df.groupby(["param_knn__weights", "param_knn__metric"]):
        g = g.sort_values("param_knn__n_neighbors")
        ax.plot(g["param_knn__n_neighbors"], g["mean_test_score"], marker="o", markersize=3,
                label=f"{weights}, {metric}")
    ax.set_xlabel("Number of neighbours k")
    ax.set_ylabel("Mean CV macro F1")
    ax.legend(fontsize=5.5)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/knn_hyperparameter_search.pdf")
    plt.close()
    print("wrote", f"{FIG_DIR}/knn_hyperparameter_search.pdf")


def plot_tree_grid():
    path = f"{RESULTS_DIR}/tree_grid_search_results.csv"
    if not os.path.exists(path):
        print("skip", path)
        return
    df = pd.read_csv(path)
    df["depth_label"] = df["param_tree__max_depth"].apply(
        lambda v: "None" if pd.isna(v) else str(int(v))
    )
    order = ["10", "20", "30", "40", "None"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, crit in zip(axes, ["gini", "entropy"]):
        sub = df[df["param_tree__criterion"] == crit]
        for leaf, g in sub.groupby("param_tree__min_samples_leaf"):
            g = g.set_index("depth_label").reindex(order).reset_index()
            ax.plot(g["depth_label"], g["mean_test_score"], marker="o", markersize=3, label=f"leaf={leaf}")
        ax.set_title(crit, fontsize=9)
        ax.set_xlabel("Maximum depth")
    axes[0].set_ylabel("Mean CV macro F1")
    axes[1].legend(fontsize=5.5)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/tree_hyperparameter_search.pdf")
    plt.close()
    print("wrote", f"{FIG_DIR}/tree_hyperparameter_search.pdf")


def comparison_table_and_figure():
    kpath = f"{RESULTS_DIR}/knn_fold_scores.csv"
    tpath = f"{RESULTS_DIR}/tree_fold_scores.csv"
    if not (os.path.exists(kpath) and os.path.exists(tpath)):
        print("skip comparison, missing fold score files")
        return
    kdf = pd.read_csv(kpath, index_col=0)
    tdf = pd.read_csv(tpath, index_col=0)

    from scipy import stats
    rows = []
    for metric in ["test_accuracy", "test_f1_macro", "test_f1_weighted", "test_balanced_accuracy"]:
        t_stat, t_p = stats.ttest_rel(tdf[metric], kdf[metric])
        diffs = tdf[metric].values - kdf[metric].values
        if (diffs == 0).all():
            w_stat, w_p = float("nan"), 1.0
        else:
            w_stat, w_p = stats.wilcoxon(tdf[metric], kdf[metric])
        rows.append({
            "metric": metric,
            "knn_mean": kdf[metric].mean(), "knn_std": kdf[metric].std(),
            "tree_mean": tdf[metric].mean(), "tree_std": tdf[metric].std(),
            "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
            "t_stat": t_stat, "t_p_value": t_p,
        })
    comp = pd.DataFrame(rows)
    comp.to_csv(f"{RESULTS_DIR}/model_comparison.csv", index=False)
    print(comp)

    metrics_disp = ["test_accuracy", "test_f1_macro", "test_f1_weighted", "test_balanced_accuracy"]
    x = np.arange(len(metrics_disp))
    width = 0.35
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.bar(x - width / 2, [kdf[m].mean() for m in metrics_disp],
           width, yerr=[kdf[m].std() for m in metrics_disp], label="kNN", color="#4C72B0", capsize=3)
    ax.bar(x + width / 2, [tdf[m].mean() for m in metrics_disp],
           width, yerr=[tdf[m].std() for m in metrics_disp], label="Tree", color="#DD8452", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Macro F1", "Weighted F1", "Balanced acc."], rotation=25, fontsize=6.5, ha="right")
    ax.set_ylabel("Score")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_comparison.pdf")
    plt.close()
    print("wrote", f"{FIG_DIR}/model_comparison.pdf")

    # per-class F1 comparison
    kcr = pd.read_csv(f"{RESULTS_DIR}/knn_classification_report.csv", index_col=0)
    tcr = pd.read_csv(f"{RESULTS_DIR}/tree_classification_report.csv", index_col=0)
    classes = [str(i) for i in range(10)]
    kf1 = [kcr.loc[c, "f1-score"] for c in classes]
    tf1 = [tcr.loc[c, "f1-score"] for c in classes]
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    x = np.arange(10)
    ax.bar(x - width / 2, kf1, width, label="kNN", color="#4C72B0")
    ax.bar(x + width / 2, tf1, width, label="Tree", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels([label_map[i] for i in range(10)], rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Per-class F1 score")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/per_class_f1.pdf")
    plt.close()
    print("wrote", f"{FIG_DIR}/per_class_f1.pdf")


if __name__ == "__main__":
    plot_confusion("tree")
    plot_confusion("knn")
    plot_feature_importance()
    plot_knn_grid()
    plot_tree_grid()
    comparison_table_and_figure()
