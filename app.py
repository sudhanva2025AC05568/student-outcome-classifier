"""
Streamlit app for the student-outcome classifiers.

Upload the test CSV, choose a model, and see how it performs.
Needs the model/ folder created by train_models.py.

Local run:  py -m streamlit run app.py
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

MODEL_FOLDER = "model"

st.set_page_config(page_title="Student Outcome Classifier",
                   page_icon="\U0001F393", layout="wide")


@st.cache_resource
def load_everything():
    # Grab every saved model, skipping the label encoder file.
    models = {}
    for file in sorted(glob.glob(os.path.join(MODEL_FOLDER, "*.joblib"))):
        key = os.path.splitext(os.path.basename(file))[0]
        if key == "target_map":
            continue
        models[key.replace("_", " ").title()] = joblib.load(file)

    # The encoder lets us turn predictions back into Dropout/Enrolled/Graduate.
    enc_file = os.path.join(MODEL_FOLDER, "target_map.joblib")
    label_map = joblib.load(enc_file) if os.path.exists(enc_file) else None
    return models, label_map


def all_metrics(y_actual, y_hat, probs, num_classes):
    # Compute the six metrics for one model.
    avg = "binary" if num_classes == 2 else "weighted"
    try:
        auc = (roc_auc_score(y_actual, probs[:, 1]) if num_classes == 2
               else roc_auc_score(y_actual, probs, multi_class="ovr", average="macro"))
    except Exception:
        auc = float("nan")

    return {
        "Accuracy": accuracy_score(y_actual, y_hat),
        "AUC": auc,
        "Precision": precision_score(y_actual, y_hat, average=avg, zero_division=0),
        "Recall": recall_score(y_actual, y_hat, average=avg, zero_division=0),
        "F1": f1_score(y_actual, y_hat, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_actual, y_hat),
    }


st.title("Student Outcome Classifier")
st.caption("Will a student drop out, stay enrolled, or graduate? Compare six models below.")

models, label_map = load_everything()
if not models:
    st.error("No models found in model/. Run train_models.py first.")
    st.stop()

# ---- sidebar controls ----
with st.sidebar:
    st.header("Controls")
    file_in = st.file_uploader("Upload test data (CSV)", type=["csv"])
    model_choice = st.selectbox("Pick a model", list(models.keys()))
    st.divider()
    st.metric("Models loaded", len(models))

if file_in is None:
    st.info("Upload test_data.csv from the sidebar to get started.")
    st.stop()

data = pd.read_csv(file_in)
st.subheader("Uploaded data")
st.write(f"{data.shape[0]} rows, {data.shape[1]} columns")
st.dataframe(data.head(10), use_container_width=True)

# let the user say which column is the answer
outcome_col = st.selectbox("Which column is the true outcome?",
                           data.columns.tolist(),
                           index=len(data.columns) - 1)

X = data.drop(columns=[outcome_col])
y_words = data[outcome_col]
y = label_map.transform(y_words) if label_map is not None else y_words.values
classes = list(label_map.classes_) if label_map is not None else sorted(np.unique(y))
num_classes = len(classes)

chosen_model = models[model_choice]

try:
    y_hat = chosen_model.predict(X)
    probs = chosen_model.predict_proba(X)
except Exception as e:
    st.error(f"Prediction failed -- do the columns match the training data?\n\n{e}")
    st.stop()

# ---- metrics for the chosen model ----
st.subheader(f"Metrics -- {model_choice}")
scores = all_metrics(y, y_hat, probs, num_classes)

metric_cols = st.columns(6)
for box, (label, val) in zip(metric_cols, scores.items()):
    box.metric(label, "n/a" if pd.isna(val) else f"{val:.3f}")

# ---- confusion matrix + report side by side ----
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Confusion matrix")
    cmatrix = confusion_matrix(y, y_hat)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cmatrix, annot=True, fmt="d", cmap="viridis", cbar=False,
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with col_b:
    st.subheader("Classification report")
    rep = classification_report(y, y_hat, target_names=[str(c) for c in classes],
                                output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).transpose().round(3), use_container_width=True)

# ---- run every model so we can compare them ----
st.subheader("All models on this test set")
table = []
for name, m in models.items():
    try:
        row = all_metrics(y, m.predict(X), m.predict_proba(X), num_classes)
        table.append({"ML Model Name": name, **row})
    except Exception:
        continue
compare_df = pd.DataFrame(table).round(4)
st.dataframe(compare_df, use_container_width=True, hide_index=True)

# ---- my own addition: a quick bar chart of F1 across models ----
st.subheader("F1 score by model")
if not compare_df.empty:
    chart_fig, chart_ax = plt.subplots(figsize=(7, 3.5))
    ordered = compare_df.sort_values("F1", ascending=True)
    chart_ax.barh(ordered["ML Model Name"], ordered["F1"], color="#4C72B0")
    chart_ax.set_xlabel("F1 score")
    chart_ax.set_xlim(0, 1)
    for i, v in enumerate(ordered["F1"]):
        chart_ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)
    st.pyplot(chart_fig)

# ---- show predictions and let the user download them ----
st.subheader("Predictions")
out = X.copy()
out["Actual"] = y_words.values
out["Predicted"] = (label_map.inverse_transform(y_hat) if label_map is not None else y_hat)
st.dataframe(out.head(50), use_container_width=True)
st.download_button("Download predictions", out.to_csv(index=False),
                   "predictions.csv", "text/csv")