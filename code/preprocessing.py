"""Shared data loading and cleaning, plus algorithm-specific preprocessing
pipelines for k-nearest neighbours and the classification tree, used for
CS741 Assignment 2 (UNSW-NB15 network traffic classification).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

DATA_PATH = "/home/john/cs741/Assignment2/assignment2/networkTraffic.csv"

NOMINAL_COLS = ["proto", "state", "service"]
TARGET_COL = "attack_cat"

# columns removed for kNN only because they are near-perfectly correlated
# duplicates of another retained column (see results/high_correlation_pairs.csv)
KNN_REDUNDANT_COLS = ["sloss", "dloss", "ct_ftp_cmd"]

# heavy-tailed, non-negative byte/rate/time features log-transformed for kNN only
KNN_LOG_COLS = [
    "dur", "sbytes", "dbytes", "rate", "sload", "dload", "spkts", "dpkts",
    "sinpkt", "dinpkt", "sjit", "djit", "smean", "dmean", "response_body_len",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "tcprtt", "synack", "ackdat",
]

RARE_PROTO_THRESHOLD = 150


def load_and_clean(path=DATA_PATH):
    """Load the raw dataset and apply the data-quality fixes that are
    independent of the downstream classification algorithm and of the
    cross-validation split. The grouping of rare protocol values is
    deliberately NOT done here: which categories count as rare must be
    learned from the training fold only (see RareCategoryGrouper), or the
    full data set would leak into the choice of which categories exist."""
    df = pd.read_csv(path, na_values=["?"], low_memory=False)
    df = df.drop(columns=["id"])
    df["service"] = df["service"].fillna("none")
    # is_ftp_login values of 2 and 4 were investigated and left uncorrected:
    # they match ct_ftp_cmd exactly on every affected row (see report), so
    # they are a genuine count rather than corrupted data.
    return df


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Groups categories occurring fewer than `threshold` times into a
    single 'other' category. The set of rare categories is learned in
    fit() from the training data only, so this transformer is safe to use
    inside a cross-validated Pipeline without leaking test-fold category
    frequencies into the training-fold representation."""

    def __init__(self, column, threshold):
        self.column = column
        self.threshold = threshold

    def fit(self, X, y=None):
        col = X[self.column]
        vc = col.value_counts()
        self.rare_categories_ = set(vc[vc < self.threshold].index)
        self.feature_names_in_ = np.asarray(X.columns)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        X = X.copy()
        X[self.column] = X[self.column].where(
            ~X[self.column].isin(self.rare_categories_), other="other"
        )
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


def stratified_sample(df, n, target_col=TARGET_COL, random_state=42):
    """Draw a stratified subsample that preserves the skewed class
    proportions of the full dataset."""
    frac = n / len(df)
    parts = [g.sample(frac=frac, random_state=random_state) for _, g in df.groupby(target_col)]
    sampled = pd.concat(parts, axis=0)
    return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def get_Xy(df, target_col=TARGET_COL):
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    return X, y


def build_knn_preprocessor(X):
    numeric_cols = [c for c in X.columns if c not in NOMINAL_COLS]
    log_cols = [c for c in KNN_LOG_COLS if c in numeric_cols]
    linear_cols = [c for c in numeric_cols if c not in log_cols]

    log_pipe = Pipeline([
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    linear_pipe = Pipeline([("scale", StandardScaler())])
    cat_pipe = OneHotEncoder(handle_unknown="ignore")

    column_transform = ColumnTransformer([
        ("log_numeric", log_pipe, log_cols),
        ("linear_numeric", linear_pipe, linear_cols),
        ("categorical", cat_pipe, NOMINAL_COLS),
    ])
    return Pipeline([
        ("group_rare_proto", RareCategoryGrouper("proto", RARE_PROTO_THRESHOLD)),
        ("column_transform", column_transform),
    ])


def build_tree_preprocessor(X):
    numeric_cols = [c for c in X.columns if c not in NOMINAL_COLS]
    cat_pipe = OneHotEncoder(handle_unknown="ignore")
    column_transform = ColumnTransformer([
        ("numeric", "passthrough", numeric_cols),
        ("categorical", cat_pipe, NOMINAL_COLS),
    ])
    return Pipeline([
        ("group_rare_proto", RareCategoryGrouper("proto", RARE_PROTO_THRESHOLD)),
        ("column_transform", column_transform),
    ])


def prepare_knn_frame(df):
    df = df.drop(columns=KNN_REDUNDANT_COLS)
    return df


if __name__ == "__main__":
    df = load_and_clean()
    print("Cleaned shape:", df.shape)
    print(df["proto"].value_counts())
    print(df["is_ftp_login"].value_counts())
    knn_df = prepare_knn_frame(df)
    print("kNN frame shape:", knn_df.shape)
    X, y = get_Xy(knn_df)
    pre = build_knn_preprocessor(X)
    Xt = pre.fit_transform(X.sample(2000, random_state=0))
    print("kNN transformed dims:", Xt.shape)

    X2, y2 = get_Xy(df)
    pre2 = build_tree_preprocessor(X2)
    Xt2 = pre2.fit_transform(X2.sample(2000, random_state=0))
    print("Tree transformed dims:", Xt2.shape)
