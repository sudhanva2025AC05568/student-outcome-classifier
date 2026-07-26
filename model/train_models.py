"""
Trains six classifiers on the UCI student-outcome dataset and reports how each one does.

Run it with:  py model/train_models.py

After it finishes you get:
  - one .joblib per model inside model/ (each bundles the preprocessing)
  - target_map.joblib  -> converts Dropout/Enrolled/Graduate <-> numbers
  - test_data.csv      -> the 20% we held back, used by the Streamlit app
  - metrics.csv        -> the score table that goes into the README
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

# ----- things I might want to change -----
CSV_PATH = "data/raw_dataset.csv"   # the file downloaded from UCI
OUTCOME_COL = "Target"              # column we are trying to predict
DROP_THESE = []                     # id-like columns to throw away, if any
TEST_FRACTION = 0.2
SEED = 42
SAVE_DIR = "model"
# -----------------------------------------


def build_preprocessing(feature_df):
    # Split columns by type: numbers get scaled, text gets one-hot encoded.
    numeric_cols = feature_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]

    # numbers: fill any gaps with the median, then put everything on the same scale
    numeric_steps = Pipeline([
        ("fill", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    # OneHotEncoder renamed an argument between sklearn versions, so try both
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_steps = Pipeline([
        ("fill", SimpleImputer(strategy="most_frequent")),
        ("encode", onehot),
    ])

    return ColumnTransformer([
        ("num", numeric_steps, numeric_cols),
        ("cat", categorical_steps, categorical_cols),
    ])


def roc_auc(y_actual, probabilities, num_classes):
    # AUC is calculated differently for 2 classes vs more than 2.
    try:
        if num_classes == 2:
            return roc_auc_score(y_actual, probabilities[:, 1])
        return roc_auc_score(y_actual, probabilities, multi_class="ovr", average="macro")
    except Exception:
        return float("nan")


def get_scores(name, model, X_test, y_test, num_classes):
    # Run the model on the test set and collect all six metrics.
    y_hat = model.predict(X_test)
    probs = model.predict_proba(X_test)
    avg = "binary" if num_classes == 2 else "weighted"

    return {
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, y_hat),
        "AUC": roc_auc(y_test, probs, num_classes),
        "Precision": precision_score(y_test, y_hat, average=avg, zero_division=0),
        "Recall": recall_score(y_test, y_hat, average=avg, zero_division=0),
        "F1": f1_score(y_test, y_hat, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_hat),
    }


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # load the data (UCI file uses ; as separator)
    student_data = pd.read_csv(CSV_PATH, sep=";")
    if DROP_THESE:
        student_data = student_data.drop(columns=[c for c in DROP_THESE if c in student_data.columns])
    student_data = student_data.dropna(subset=[OUTCOME_COL])

    print(f"Loaded {student_data.shape[0]} rows x {student_data.shape[1] - 1} features")
    if student_data.shape[0] < 500 or (student_data.shape[1] - 1) < 12:
        print("WARNING: needs >= 500 instances and >= 12 features.")

    # separate features from the outcome, and turn the text outcome into numbers
    X = student_data.drop(columns=[OUTCOME_COL])
    y_text = student_data[OUTCOME_COL]

    label_map = LabelEncoder()
    y = label_map.fit_transform(y_text)
    num_classes = len(label_map.classes_)
    print(f"Outcome classes ({num_classes}): {list(label_map.classes_)}")

    # keep the class ratio the same in train and test
    strat = y if np.min(np.bincount(y)) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=SEED, stratify=strat
    )

    # save the test slice with readable labels so the app can use it
    test_slice = X_test.copy()
    test_slice[OUTCOME_COL] = label_map.inverse_transform(y_test)
    test_slice.to_csv("test_data.csv", index=False)
    print(f"Wrote test_data.csv ({len(test_slice)} rows)")

    # the six classifiers I'm comparing
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=SEED),
        "kNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    }

    results = []
    for name, model in models.items():
        # glue preprocessing and the model together so they travel as one unit
        full_model = Pipeline([
            ("preprocess", build_preprocessing(X_train)),
            ("classifier", model),
        ])
        full_model.fit(X_train, y_train)

        results.append(get_scores(name, full_model, X_test, y_test, num_classes))

        file_name = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(full_model, os.path.join(SAVE_DIR, file_name))
        print(f"  trained + saved: {file_name}")

    # the encoder is needed by the app to decode predictions back to words
    joblib.dump(label_map, os.path.join(SAVE_DIR, "target_map.joblib"))

    scores_df = pd.DataFrame(results).round(4)
    scores_df.to_csv("metrics.csv", index=False)

    print("\n=== Comparison Table (paste into README.md) ===")
    print(scores_df.to_markdown(index=False))
    winner = scores_df.loc[scores_df["F1"].idxmax(), "ML Model Name"]
    print(f"\nHighest F1: {winner}")


if __name__ == "__main__":
    main()