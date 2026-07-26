"""
Student outcome classification - model training and evaluation.

Workflow:
    1. Place the dataset at data/raw_dataset.csv
    2. Adjust the settings block (source_file, label_column)
    3. Run:  py model/train_models.py

Produces:
    model/*.joblib          fitted pipelines (preprocessing bundled with each classifier)
    model/target_map.joblib label encoder for the outcome column
    test_data.csv           hold-out slice used by the Streamlit app
    metrics.csv             score summary for the README
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

# ------------------------------- settings -------------------------------
source_file = "data/raw_dataset.csv"   # downloaded UCI file
label_column = "Target"                # outcome column in the CSV
columns_to_remove = []                 # any id-style columns to discard
holdout_fraction = 0.2
seed = 42
output_folder = "model"
# ------------------------------------------------------------------------


def make_feature_pipeline(features_frame):
    """Numeric columns: median-fill then standardise.
       Categorical columns: mode-fill then one-hot."""
    number_columns = features_frame.select_dtypes(include=np.number).columns.tolist()
    text_columns = [col for col in features_frame.columns if col not in number_columns]

    number_branch = Pipeline([
        ("fill_gaps", SimpleImputer(strategy="median")),
        ("standardise", StandardScaler()),
    ])

    # argument name changed across scikit-learn versions
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    text_branch = Pipeline([
        ("fill_gaps", SimpleImputer(strategy="most_frequent")),
        ("one_hot", encoder),
    ])

    return ColumnTransformer([
        ("numbers", number_branch, number_columns),
        ("categories", text_branch, text_columns),
    ])


def compute_auc(truth, class_probs, class_count):
    """Area under ROC, handling binary and multiclass separately."""
    try:
        if class_count == 2:
            return roc_auc_score(truth, class_probs[:, 1])
        return roc_auc_score(truth, class_probs, multi_class="ovr", average="macro")
    except Exception:
        return float("nan")


def score_model(model_label, fitted_pipeline, features_test, truth_test, class_count):
    predictions = fitted_pipeline.predict(features_test)
    class_probs = fitted_pipeline.predict_proba(features_test)
    averaging = "binary" if class_count == 2 else "weighted"

    return {
        "ML Model Name": model_label,
        "Accuracy": accuracy_score(truth_test, predictions),
        "AUC": compute_auc(truth_test, class_probs, class_count),
        "Precision": precision_score(truth_test, predictions, average=averaging, zero_division=0),
        "Recall": recall_score(truth_test, predictions, average=averaging, zero_division=0),
        "F1": f1_score(truth_test, predictions, average=averaging, zero_division=0),
        "MCC": matthews_corrcoef(truth_test, predictions),
    }


def run():
    os.makedirs(output_folder, exist_ok=True)

    dataset = pd.read_csv(source_file, sep=";")
    if columns_to_remove:
        dataset = dataset.drop(columns=[c for c in columns_to_remove if c in dataset.columns])
    dataset = dataset.dropna(subset=[label_column])

    print(f"Loaded {dataset.shape[0]} rows x {dataset.shape[1] - 1} features")
    if dataset.shape[0] < 500 or (dataset.shape[1] - 1) < 12:
        print("WARNING: needs >= 500 instances and >= 12 features.")

    features = dataset.drop(columns=[label_column])
    outcome_text = dataset[label_column]

    target_map = LabelEncoder()
    outcome = target_map.fit_transform(outcome_text)
    class_count = len(target_map.classes_)
    print(f"Outcome classes ({class_count}): {list(target_map.classes_)}")

    keep_ratio = outcome if np.min(np.bincount(outcome)) >= 2 else None
    features_train, features_test, outcome_train, outcome_test = train_test_split(
        features, outcome, test_size=holdout_fraction, random_state=seed, stratify=keep_ratio
    )

    # Persist the hold-out slice (original text labels) for the app.
    holdout = features_test.copy()
    holdout[label_column] = target_map.inverse_transform(outcome_test)
    holdout.to_csv("test_data.csv", index=False)
    print(f"Wrote test_data.csv ({len(holdout)} rows)")

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=seed),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=seed),
        "kNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=seed),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
    }

    summary = []
    for model_label, estimator in classifiers.items():
        pipeline = Pipeline([
            ("prepare", make_feature_pipeline(features_train)),
            ("model", estimator),
        ])
        pipeline.fit(features_train, outcome_train)

        summary.append(score_model(model_label, pipeline, features_test, outcome_test, class_count))

        saved_name = model_label.lower().replace(" ", "_") + ".joblib"
        joblib.dump(pipeline, os.path.join(output_folder, saved_name))
        print(f"  trained + saved: {saved_name}")

    joblib.dump(target_map, os.path.join(output_folder, "target_map.joblib"))

    score_table = pd.DataFrame(summary).round(4)
    score_table.to_csv("metrics.csv", index=False)

    print("\n=== Comparison Table (paste into README.md) ===")
    print(score_table.to_markdown(index=False))
    top = score_table.loc[score_table["F1"].idxmax(), "ML Model Name"]
    print(f"\nHighest F1: {top}")


if __name__ == "__main__":
    run()