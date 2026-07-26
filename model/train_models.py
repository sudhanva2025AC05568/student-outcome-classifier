"""
ML Assignment 2 -- train and evaluate classification models.

HOW TO USE
1. Put your dataset CSV somewhere (e.g. data/raw_dataset.csv)
2. Edit the CONFIG block below (DATA_PATH, TARGET_COL, DROP_COLS)
3. Run:  python train_models.py

Outputs:
  model/<model_name>.joblib   -- full pipeline (preprocessing + classifier)
  model/label_encoder.joblib  -- target label encoder
  test_data.csv               -- held-out test split, for the Streamlit app
  metrics.csv                 -- comparison table for your README
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ----------------------------- CONFIG -----------------------------
DATA_PATH = "data/raw_dataset.csv"   # your UCI file
TARGET_COL = "Target"                # <-- was "target", must be capital T
DROP_COLS = []
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_DIR = "model"
# ------------------------------------------------------------------


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Impute + scale numeric columns, impute + one-hot encode categorical ones."""
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    # sparse_output was named `sparse` before scikit-learn 1.2
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", ohe),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])


def auc_score(y_true, proba, n_classes: int) -> float:
    """ROC-AUC for both the binary and the multi-class case."""
    try:
        if n_classes == 2:
            return roc_auc_score(y_true, proba[:, 1])
        return roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
    except Exception:
        return float("nan")


def evaluate(name, pipe, X_test, y_test, n_classes):
    y_pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)
    average = "binary" if n_classes == 2 else "weighted"

    return {
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc_score(y_test, proba, n_classes),
        "Precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_test, y_pred, average=average, zero_division=0),
        "F1": f1_score(y_test, y_pred, average=average, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH, sep=";")
    if DROP_COLS:
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.dropna(subset=[TARGET_COL])

    print(f"Loaded {df.shape[0]} rows x {df.shape[1] - 1} features")
    if df.shape[0] < 500 or (df.shape[1] - 1) < 12:
        print("WARNING: assignment requires >= 500 instances and >= 12 features.")

    X = df.drop(columns=[TARGET_COL])
    y_raw = df[TARGET_COL]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    n_classes = len(label_encoder.classes_)
    print(f"Target classes ({n_classes}): {list(label_encoder.classes_)}")

    stratify = y if np.min(np.bincount(y)) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )

    # Save the held-out split for the Streamlit app (original labels, not encoded)
    test_export = X_test.copy()
    test_export[TARGET_COL] = label_encoder.inverse_transform(y_test)
    test_export.to_csv("test_data.csv", index=False)
    print(f"Wrote test_data.csv ({len(test_export)} rows)")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    rows = []
    for name, clf in models.items():
        pipe = Pipeline([("prep", build_preprocessor(X_train)), ("clf", clf)])
        pipe.fit(X_train, y_train)

        rows.append(evaluate(name, pipe, X_test, y_test, n_classes))

        fname = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(pipe, os.path.join(MODEL_DIR, fname))
        print(f"  trained + saved: {fname}")

    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.joblib"))

    results = pd.DataFrame(rows).round(4)
    results.to_csv("metrics.csv", index=False)

    print("\n=== Comparison Table (paste into README.md) ===")
    print(results.to_markdown(index=False))
    best = results.loc[results["F1"].idxmax(), "ML Model Name"]
    print(f"\nHighest F1: {best}")


if __name__ == "__main__":
    main()
