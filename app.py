"""
ML Assignment 2 -- Streamlit front-end.

Run locally:  streamlit run app.py
Expects the model/ folder produced by train_models.py.
"""

import glob
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

MODEL_DIR = "model"

st.set_page_config(page_title="Classifier Comparison", page_icon="📊", layout="wide")


@st.cache_resource
def load_models():
    """Load every saved pipeline plus the target label encoder."""
    pipelines = {}
    for path in sorted(glob.glob(os.path.join(MODEL_DIR, "*.joblib"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name == "label_encoder":
            continue
        pipelines[name.replace("_", " ").title()] = joblib.load(path)

    enc_path = os.path.join(MODEL_DIR, "label_encoder.joblib")
    encoder = joblib.load(enc_path) if os.path.exists(enc_path) else None
    return pipelines, encoder


def compute_metrics(y_true, y_pred, proba, n_classes):
    average = "binary" if n_classes == 2 else "weighted"
    try:
        auc = (roc_auc_score(y_true, proba[:, 1]) if n_classes == 2
               else roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    except Exception:
        auc = float("nan")

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


st.title("Classification Model Explorer")
st.caption("Upload your test CSV, pick a model, and inspect its performance.")

pipelines, encoder = load_models()
if not pipelines:
    st.error("No models found in model/. Run train_models.py first.")
    st.stop()

# ---------------------------- Sidebar ----------------------------
with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Test data (CSV)", type=["csv"])
    chosen = st.selectbox("Model", list(pipelines.keys()))
    st.divider()
    st.write("Models available:", len(pipelines))

if uploaded is None:
    st.info("Upload a CSV in the sidebar to begin. Use the `test_data.csv` from the repo.")
    st.stop()

df = pd.read_csv(uploaded)
st.subheader("Uploaded data")
st.write(f"{df.shape[0]} rows, {df.shape[1]} columns")
st.dataframe(df.head(10), use_container_width=True)

target_col = st.selectbox(
    "Which column holds the true label?",
    df.columns.tolist(),
    index=len(df.columns) - 1,
)

X = df.drop(columns=[target_col])
y_true_raw = df[target_col]
y_true = encoder.transform(y_true_raw) if encoder is not None else y_true_raw.values
class_names = list(encoder.classes_) if encoder is not None else sorted(np.unique(y_true))
n_classes = len(class_names)

pipe = pipelines[chosen]

try:
    y_pred = pipe.predict(X)
    proba = pipe.predict_proba(X)
except Exception as err:
    st.error(f"Prediction failed -- do the uploaded columns match the training data?\n\n{err}")
    st.stop()

# ---------------------------- Metrics ----------------------------
st.subheader(f"Evaluation metrics -- {chosen}")
metrics = compute_metrics(y_true, y_pred, proba, n_classes)

cols = st.columns(6)
for col, (label, value) in zip(cols, metrics.items()):
    col.metric(label, "n/a" if pd.isna(value) else f"{value:.3f}")

# --------------------- Confusion matrix + report -------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.subheader("Classification report")
    report = classification_report(
        y_true, y_pred, target_names=[str(c) for c in class_names],
        output_dict=True, zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

# ------------------------ All-model comparison ---------------------
st.subheader("All models on this test set")
rows = []
for name, p in pipelines.items():
    try:
        m = compute_metrics(y_true, p.predict(X), p.predict_proba(X), n_classes)
        rows.append({"ML Model Name": name, **m})
    except Exception:
        continue

comparison = pd.DataFrame(rows).round(4)
st.dataframe(comparison, use_container_width=True, hide_index=True)

st.subheader("Predictions")
preview = X.copy()
preview["Actual"] = y_true_raw.values
preview["Predicted"] = (encoder.inverse_transform(y_pred) if encoder is not None else y_pred)
st.dataframe(preview.head(50), use_container_width=True)
st.download_button("Download predictions", preview.to_csv(index=False),
                   "predictions.csv", "text/csv")