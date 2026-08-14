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
    models = {}
    for file in sorted(glob.glob(os.path.join(MODEL_FOLDER, "*.joblib"))):
        key = os.path.splitext(os.path.basename(file))[0]
        if key == "target_map":
            continue
        nice_name = key.replace("_", " ").title().replace("Knn", "kNN")
        models[nice_name] = joblib.load(file)

    enc_file = os.path.join(MODEL_FOLDER, "target_map.joblib")
    label_map = joblib.load(enc_file) if os.path.exists(enc_file) else None
    return models, label_map


def all_metrics(y_actual, y_hat, probs, num_classes):
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


st.markdown("<h1 style='color:#4C72B0;'>🎓 Student Outcome Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:16px; color:#555;'>Will a student drop out, stay enrolled, or graduate? Compare six models below.</p>", unsafe_allow_html=True)
st.markdown("<p style='color:#2E8B57; font-weight:bold;'>Six supervised models • UCI Student Dropout Dataset (ID 697) • 36 features • 3 classes</p>", unsafe_allow_html=True)
st.markdown("<p style='color:#888; font-style:italic;'>by SUDHANVA S A (2025AC05568) — M.Tech AIML, Assignment 2</p>", unsafe_allow_html=True)

models, label_map = load_everything()
if not models:
    st.error("No models found in model/. Run train_models.py first.")
    st.stop()

# ---- sidebar controls ----
model_names = list(models.keys())

# applied model = what results are shown for; only changes when Apply is clicked
if "applied_model" not in st.session_state:
    st.session_state.applied_model = model_names[0]

with st.sidebar:
    st.header("Controls")
    file_in = st.file_uploader("Upload test data (CSV)", type=["csv"])

    with st.form("model_selection"):
        st.selectbox(
            "Pick a model",
            model_names,
            index=model_names.index(st.session_state.applied_model),
            key="model_picker",
        )
        apply_model = st.form_submit_button("Apply Model")
        st.caption("Click **Apply Model** to load results")

    if apply_model:
        st.session_state.applied_model = st.session_state.model_picker

    st.divider()
    st.metric("Models loaded", len(models))

if file_in is None:
    st.info("Upload test_data.csv from the sidebar to get started.")
    st.stop()

data = pd.read_csv(file_in)
st.markdown(f"## Current model: `{st.session_state.applied_model}`")
st.subheader("Uploaded data")
st.write(f"**Test dataset:** {data.shape[0]} rows × {data.shape[1]} columns (36 features + 1 target)")
st.dataframe(data.head(10), use_container_width=True)

outcome_col = "Target" if "Target" in data.columns else data.columns[-1]

X = data.drop(columns=[outcome_col])
y_words = data[outcome_col]
try:
    y = label_map.transform(y_words) if label_map is not None else y_words.values
except ValueError:
    st.error("Selected column doesn't look like the outcome. Please select **Target**.")
    st.stop()
classes = list(label_map.classes_) if label_map is not None else sorted(np.unique(y))
num_classes = len(classes)

model_choice = st.session_state.applied_model
chosen_model = models[model_choice]

try:
    y_hat = chosen_model.predict(X)
    probs = chosen_model.predict_proba(X)
except Exception as e:
    st.error(f"Prediction failed -- do the columns match the training data?\n\n{e}")
    st.stop()

st.subheader(f"Evaluation Metrics -- {model_choice}")
scores = all_metrics(y, y_hat, probs, num_classes)

metric_cols = st.columns(6)
for box, (label, val) in zip(metric_cols, scores.items()):
    box.metric(label, "n/a" if pd.isna(val) else f"{val:.3f}")

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

st.subheader("Comparison of all models on the test set")
table = []
for name, m in models.items():
    try:
        row = all_metrics(y, m.predict(X), m.predict_proba(X), num_classes)
        table.append({"ML Model Name": name, **row})
    except Exception:
        continue
compare_df = pd.DataFrame(table).round(4)
st.dataframe(compare_df, use_container_width=True, hide_index=True)

if not compare_df.empty:
    best = compare_df.loc[compare_df["F1"].idxmax()]
    st.success(f"🏆 Best on this data: **{best['ML Model Name']}** (F1 = {best['F1']:.4f}, MCC = {best['MCC']:.4f})")

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

st.subheader("Predictions vs Actual")
out = pd.DataFrame({
    "Actual": y_words.values,
    "Predicted": label_map.inverse_transform(y_hat) if label_map is not None else y_hat,
})
match_bool = out["Actual"] == out["Predicted"]
out["Match"] = np.where(match_bool, "✅", "❌")

correct = int(match_bool.sum())
total = len(out)
st.write(f"**Correctly predicted:** {correct} / {total} ({correct/total:.1%})")
st.dataframe(out.head(50), use_container_width=True)