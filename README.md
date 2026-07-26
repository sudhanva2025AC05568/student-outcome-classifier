# Student Outcome Classifier

M.Tech (AIML/DSE) — Machine Learning — Assignment 2
**Name:** SUDHANVA S A  |  **BITS ID:** 2025AC05568

- **Live Streamlit app:** https://student-outcome-classifier-gyeofgb4xk59585mm4pvhm.streamlit.app/
- **GitHub repository:** https://github.com/sudhanva2025AC05568/student-outcome-classifier

---

## a. Problem statement

The goal of this project is to predict a student's final academic outcome — **Dropout**, **Enrolled**, or **Graduate** — using information known about the student at enrolment and after their first two semesters. This is a **multi-class classification** problem with three classes.

> ✍️ WRITE 1–2 LINES IN YOUR OWN WORDS: why this problem interested you / why it is useful.
> Example: "I chose this problem because identifying at-risk students early could help colleges step in before a student drops out."

## b. Dataset description  *(1 mark)*

| Item | Detail |
|---|---|
| Source | UCI Machine Learning Repository, dataset ID 697 (Realinho et al., 2021) |
| URL | https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success |
| Instances | 4,424  (meets the ≥ 500 requirement) |
| Features | 36  (meets the ≥ 12 requirement) |
| Target column | `Target` → Dropout / Enrolled / Graduate |
| Problem type | Multi-class classification (3 classes) |
| Class balance | Graduate ≈ 50%, Dropout ≈ 32%, Enrolled ≈ 18% (imbalanced) |
| Missing values | None in the raw file; the pipeline still imputes as a safeguard |
| Feature types | Mix of numeric (grades, ages, economic indicators) and categorical (course, marital status, nationality) |

The features fall into three groups: demographic and socio-economic background, the academic path (course, attendance mode, prior qualification), and first- and second-semester performance (units enrolled, units approved, and grades). The semester-performance features turned out to be the most predictive.

## c. GitHub repository link  *(1 mark)*

https://github.com/sudhanva2025AC05568/student-outcome-classifier

The repository contains: `app.py`, `requirements.txt`, `README.md`, `test_data.csv`, and a `model/` folder with the training script (`train_models.py`) and all saved model files.

## d. Models used  *(5 marks: metrics · 3 marks: observations)*

All six models were trained on the **same** dataset using an 80/20 stratified split (`random_state=42`) and an identical preprocessing pipeline: median imputation + standardisation for numeric features, and most-frequent imputation + one-hot encoding for categorical features. Preprocessing is bundled inside each model's pipeline, so the test data is transformed using statistics learned only from the training data.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7684 | 0.8778 | 0.7500 | 0.7684 | 0.7531 | 0.6150 |
| Decision Tree | 0.7277 | 0.8115 | 0.7261 | 0.7277 | 0.7242 | 0.5527 |
| kNN | 0.6734 | 0.7883 | 0.6530 | 0.6734 | 0.6576 | 0.4529 |
| Naive Bayes | 0.6588 | 0.7893 | 0.6334 | 0.6588 | 0.6417 | 0.4279 |
| Random Forest | 0.7718 | 0.8875 | 0.7575 | 0.7718 | 0.7569 | 0.6208 |
| Gradient Boosting | 0.7593 | 0.8892 | 0.7484 | 0.7593 | 0.7504 | 0.6011 |

### Observations

> ✍️ REWRITE EACH OBSERVATION IN YOUR OWN WORDS. The notes below are a guide — say them the way you would explain them. This is a 3-mark section, so make it sound like you.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Did surprisingly well for a simple linear model — second-highest F1 and MCC. Suggests the classes are largely linearly separable. Fast and easy to interpret. |
| Decision Tree | Middle of the pack. A single tree captures some non-linear patterns but tends to overfit; limiting depth helped, but it still trailed the ensemble models. |
| kNN | One of the weakest. One-hot encoding pushes the data into a high-dimensional space where distance-based voting becomes less reliable, and the small Enrolled class gets outvoted. |
| Naive Bayes | Lowest scores overall. Its assumption that features are independent is violated here (grades and units are correlated), which hurts it — though its AUC shows it still ranks students reasonably. |
| Random Forest | Best overall on Accuracy, F1, and MCC. Averaging many de-correlated trees fixes the overfitting that hurt the single tree, and it handles the mixed feature types well. |
| Gradient Boosting | Almost as good as Random Forest and had the highest AUC of all. Builds trees sequentially to correct earlier errors; slightly lower F1 but an excellent second ensemble. |
| **Overall winner for this dataset** | **Random Forest** — highest F1 (0.757) and MCC (0.621). MCC matters most here because the dataset is imbalanced, and Random Forest leads on it. Gradient Boosting is a very close runner-up. |

Every model comfortably beat the majority-class baseline (~50% accuracy), which confirms they all learned real structure rather than just guessing the biggest class. Across all models the confusion matrix shows the same pattern: Graduate and Dropout are predicted well, but the smaller, more ambiguous **Enrolled** class is the hardest — those students are mid-programme and could still go either way.

---

## Challenges faced

> ✍️ THIS SECTION IS UNIQUELY YOURS — keep it, it proves the work is yours.
> Example: "Deployment on Streamlit Cloud initially failed because the models were trained on a newer scikit-learn version than the one Cloud installed, which caused a version-mismatch error. I fixed it by pinning scikit-learn to the matching version in requirements.txt. I also learned that Streamlit Cloud's Python version has to be set in Advanced settings, not via runtime.txt."

## Repository structure

```
student-outcome-classifier/
├── app.py                 # Streamlit application
├── requirements.txt
├── README.md
├── test_data.csv          # 20% held-out test split (885 rows)
├── metrics.csv            # generated comparison table
└── model/
    ├── train_models.py    # training + evaluation script
    ├── *.joblib           # saved pipelines for all six models
    └── target_map.joblib  # outcome label encoder
```

## How to run locally

```bash
pip install -r requirements.txt
py model/train_models.py     # creates model/, test_data.csv, metrics.csv
py -m streamlit run app.py
```

## Streamlit app features

- CSV upload for test data
- Model selection dropdown
- All six evaluation metrics shown per model
- Confusion matrix heatmap and classification report
- Comparison of every model on the uploaded test set
- F1-score bar chart across models
- Downloadable predictions
