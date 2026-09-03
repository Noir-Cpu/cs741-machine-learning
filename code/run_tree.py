"""Hyperparameter tuning and k-fold cross-validated evaluation of the
classification tree on the pre-processed network traffic data."""
import sys, time, json
sys.path.insert(0, "/home/john/cs741/Assignment2/code")
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV, cross_validate, cross_val_predict,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, classification_report,
)

from preprocessing import load_and_clean, get_Xy, build_tree_preprocessor

RESULTS_DIR = "/home/john/cs741/Assignment2/results"
RANDOM_STATE = 42
N_FOLDS = 5
N_REPEATS = 3  # repeats used only for the final fold-score comparison, not for the grid search

def main():
    t0 = time.time()
    df = load_and_clean()
    X, y = get_Xy(df)
    print("Data shape:", X.shape, "loaded in", round(time.time() - t0, 1), "s")

    pre = build_tree_preprocessor(X)
    pipe = Pipeline([("pre", pre), ("tree", DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE))])

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    param_grid = {
        "tree__criterion": ["gini", "entropy"],
        "tree__max_depth": [10, 20, 30, 40, None],
        "tree__min_samples_leaf": [1, 5, 20, 50],
    }

    print("Starting grid search ...")
    t0 = time.time()
    gs = GridSearchCV(pipe, param_grid, scoring="f1_macro", cv=cv, n_jobs=-1, verbose=2, refit=False)
    gs.fit(X, y)
    print("Grid search finished in", round(time.time() - t0, 1), "s")

    cvres = pd.DataFrame(gs.cv_results_)
    cvres.to_csv(f"{RESULTS_DIR}/tree_grid_search_results.csv", index=False)
    best_params = gs.best_params_
    print("Best params:", best_params, "best macro-F1:", gs.best_score_)

    with open(f"{RESULTS_DIR}/tree_best_params.json", "w") as f:
        json.dump({"best_params": best_params, "best_cv_macro_f1": gs.best_score_}, f, indent=2)

    # ---- final k-fold CV evaluation with the selected configuration ----
    best_tree = DecisionTreeClassifier(
        criterion=best_params["tree__criterion"],
        max_depth=best_params["tree__max_depth"],
        min_samples_leaf=best_params["tree__min_samples_leaf"],
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    final_pipe = Pipeline([("pre", build_tree_preprocessor(X)), ("tree", best_tree)])

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
        "balanced_accuracy": "balanced_accuracy",
    }
    print("Running final repeated cross_validate ...")
    t0 = time.time()
    cv_repeated = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    cvout = cross_validate(final_pipe, X, y, cv=cv_repeated, scoring=scoring, n_jobs=-1, return_train_score=False)
    print("Final CV finished in", round(time.time() - t0, 1), "s")
    fold_scores = pd.DataFrame({k: v for k, v in cvout.items() if k.startswith("test_")})
    fold_scores.index.name = "fold"
    fold_scores.to_csv(f"{RESULTS_DIR}/tree_fold_scores.csv")
    print(fold_scores)
    print(fold_scores.mean())
    print(fold_scores.std())

    # ---- out-of-fold predictions for confusion matrix and per-class report ----
    print("Running cross_val_predict for confusion matrix ...")
    t0 = time.time()
    y_pred = cross_val_predict(final_pipe, X, y, cv=cv, n_jobs=-1)
    print("cross_val_predict finished in", round(time.time() - t0, 1), "s")
    cm = confusion_matrix(y, y_pred)
    np.savetxt(f"{RESULTS_DIR}/tree_confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    report = classification_report(y, y_pred, output_dict=True)
    pd.DataFrame(report).T.to_csv(f"{RESULTS_DIR}/tree_classification_report.csv")

    # ---- feature importances from a full-data refit for reporting ----
    final_pipe.fit(X, y)
    feat_names = final_pipe.named_steps["pre"].get_feature_names_out()
    importances = final_pipe.named_steps["tree"].feature_importances_
    imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values(
        "importance", ascending=False
    )
    imp_df.to_csv(f"{RESULTS_DIR}/tree_feature_importances.csv", index=False)

    summary = {
        "best_params": best_params,
        "fold_mean": fold_scores.mean().to_dict(),
        "fold_std": fold_scores.std().to_dict(),
        "overall_accuracy": accuracy_score(y, y_pred),
        "overall_macro_f1": f1_score(y, y_pred, average="macro"),
        "overall_balanced_accuracy": balanced_accuracy_score(y, y_pred),
    }
    with open(f"{RESULTS_DIR}/tree_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
