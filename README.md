# Student Outcome Classifier

M.Tech (AIML/DSE) — Machine Learning — Assignment 2
Name: <your name> | BITS ID: <your ID>

Live app: https://student-outcome-classifier-gyeofgb4xk59585mm4pvhm.streamlit.app/
Repository: https://github.com/sudhanva2025AC05568/student-outcome-classifier

---

## a. Problem statement

The task is to predict a student's academic outcome — **Dropout**, **Enrolled**, or **Graduate** — from information available about the student at enrolment and after the first two semesters (demographics, prior qualifications, course details, and academic performance). This is a **multi-class classification** problem with three outcome classes. Early identification of students at risk of dropping out lets institutions intervene with support before the student leaves.

## b. Dataset description

| Item | Detail |
|---|---|
| Source | UCI Machine Learning Repository (ID 697), Realinho et al., 2021 |
| URL | https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success |
| Instances | 4,424 |
| Features | 36 |
| Target | `Target` — Dropout / Enrolled / Graduate |
| Class balance | Graduate ~50%, Dropout ~32%, Enrolled ~18% (imbalanced) |
| Missing values | None in the raw file; pipeline still imputes defensively |
| Feature types | Mix of numeric (grades, ages, economic indicators) and categorical (course, marital status, nationality) |

The features cover three broad groups: demographic and socio-economic background, details of the academic path (course, attendance mode, prior qualification), and first- and second-semester performance (units enrolled, approved, and grades). The semester-performance features are the most informative for the outcome.

## c. GitHub repository link

https://github.com/sudhanva2025AC05568/student-outcome-classifier

## d. Models used

All six models were trained on an identical 80/20 stratified split with the same preprocessing pipeline: median imputation and standardisation for numeric features, most-frequent imputation and one-hot encoding for categorical features. Preprocessing is bundled inside each model's pipeline so the test data is transformed with statistics learned only from the training data.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7684 | 0.8778 | 0.7500 | 0.7684 | 0.7531 | 0.6150 |
| Decision Tree | 0.7277 | 0.8115 | 0.7261 | 0.7277 | 0.7242 | 0.5527 |
| kNN | 0.6734 | 0.7883 | 0.6530 | 0.6734 | 0.6576 | 0.4529 |
| Naive Bayes | 0.6588 | 0.7893 | 0.6334 | 0.6588 | 0.6417 | 0.4279 |
| Random Forest (Ensemble) | 0.7718 | 0.8875 | 0.7575 | 0.7718 | 0.7569 | 0.6208 |
| Gradient Boosting (Ensemble) | 0.7593 | 0.8892 | 0.7484 | 0.7593 | 0.7504 | 0.6011 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Surprisingly strong for a linear model — second-best F1 and MCC. The classes are largely linearly separable in this feature space, so the simple decision boundary works well. Fast and interpretable. |
| Decision Tree | Middle of the pack. A single tree captures some non-linear structure but is prone to overfitting; capping depth at 8 kept it reasonable but it still trails the ensembles that average many trees. |
| kNN | Weakest but one. Distance-based voting suffers here because one-hot encoding pushes the data into a high-dimensional space where "nearest" becomes less meaningful, and the minority Enrolled class is easily outvoted by neighbours. |
| Naive Bayes | Lowest scores. Its assumption that features are independent given the class is badly violated — semester grades, units approved, and prior qualification are strongly correlated — so its probability estimates are off, though its AUC shows it still ranks cases reasonably. |
| Random Forest (Ensemble) | Best overall on accuracy, F1, and MCC. Bagging many de-correlated trees controls the variance that hurt the single tree, and it handles the mixed feature types well. |
| Gradient Boosting (Ensemble) | Very close to Random Forest and the highest AUC of all, meaning it ranks students by risk slightly better. Marginally lower F1, but an excellent second ensemble. |
| **Overall winner** | **Random Forest** — highest F1 (0.757) and MCC (0.621). MCC matters most here because the dataset is imbalanced, and Random Forest leads on it. Gradient Boosting is a close runner-up with the best AUC. |

All six models comfortably beat the majority-class baseline (~50% accuracy), confirming they learned real structure. The confusion matrix shows every model handles Graduate and Dropout well but struggles with the smaller, more ambiguous Enrolled class — students mid-programme who could still go either way.

---

## Repository structure

```
student-outcome-classifier/
├── app.py                 # Streamlit application
├── requirements.txt
├── README.md
├── test_data.csv          # Held-out test split
├── metrics.csv            # Generated comparison table
└── model/
    ├── train_models.py    # Training + evaluation script
    ├── *.joblib           # Saved pipelines for all models
    └── target_map.joblib  # Outcome label encoder
```

## How to reproduce

```bash
pip install -r requirements.txt
py model/train_models.py     # writes model/, test_data.csv, metrics.csv
py -m streamlit run app.py
```

## Streamlit app features

- CSV upload for test data
- Model selection dropdown
- Six evaluation metrics displayed per model
- Confusion matrix heatmap and classification report
- Comparison of all models on the uploaded test set
- Downloadable predictions
