# Student Outcome Classifier

M.Tech (AIML/DSE) — Machine Learning — Assignment 2
**Name:** SUDHANVA S A  |  **BITS ID:** 2025AC05568

- **Live Streamlit app:** https://student-outcome-classifier-gyeofgb4xk59585mm4pvhm.streamlit.app/
- **GitHub repository:** https://github.com/sudhanva2025AC05568/student-outcome-classifier

---

## a. Problem statement

This project predicts whether a student will end up as a **Dropout**, stay **Enrolled**, or **Graduate**, based on details known about them at admission and after their first two semesters. It is a multi-class classification problem with three classes.

I picked this problem because student dropout is something colleges genuinely care about, and if a model can flag the students who are likely to leave, the college gets a chance to help them before it happens. That practical angle made it more interesting to me than a generic dataset.

## b. Dataset description

| Item | Detail |
|---|---|
| Source | UCI Machine Learning Repository, dataset ID 697 (Realinho et al., 2021) |
| URL | https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success |
| Instances | 4,424 |
| Features | 36 |
| Target column | `Target` → Dropout / Enrolled / Graduate |
| Problem type | Multi-class classification (3 classes) |
| Class balance | Graduate ≈ 50%, Dropout ≈ 32%, Enrolled ≈ 18% (imbalanced) |
| Missing values | None in the raw file; pipeline imputes anyway as a safeguard |
| Feature types | Numeric (grades, ages, economic indicators) and categorical (course, marital status, nationality) |

The columns cover the student's background, their academic path, and their first- and second-semester results. In practice the semester grades and units-passed columns carried most of the predictive signal.

## c. GitHub repository link

https://github.com/sudhanva2025AC05568/student-outcome-classifier

Contains `app.py`, `requirements.txt`, `README.md`, `test_data.csv`, and a `model/` folder with the training script and all saved models.

## d. Models used

All six models use the same 80/20 stratified split (`random_state=42`) and the same preprocessing: median fill + scaling for numeric columns, most-frequent fill + one-hot encoding for categorical ones. The preprocessing is kept inside each model's pipeline so the test set is only ever transformed with values learned from the training set.

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

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Did better than I expected for such a simple model — came second on F1 and MCC. Looks like the classes can be separated fairly well with a straight boundary. |
| Decision Tree | Okay but not great. A single tree picks up some patterns but overfits, so even after limiting its depth it stayed behind the ensemble models. |
| kNN | One of the two weakest. After one-hot encoding there are a lot of columns, and distance-based methods don't work as well in that many dimensions. The small Enrolled class also gets outvoted by its neighbours. |
| Naive Bayes | Weakest overall. It assumes the features are independent, which isn't true here since grades and units passed clearly move together, so its predictions suffer. Its AUC is still okay though. |
| Random Forest | Best model for this dataset — top on Accuracy, F1 and MCC. Averaging many trees cancels out the overfitting a single tree had. |
| Gradient Boosting | Nearly tied with Random Forest and actually had the best AUC. Its F1 was slightly lower, so I kept Random Forest as the winner. |
| **Overall winner for this dataset** | **Random Forest** — highest F1 (0.757) and MCC (0.621). I went by MCC because the classes are imbalanced and MCC handles that better than accuracy. |

One thing common to every model: they all handle Graduate and Dropout well but struggle with the Enrolled class. That makes sense to me, since an "Enrolled" student is still mid-course and could realistically end up in either of the other two groups.

## Challenges faced

The part that took the most time was deployment. Everything worked on my laptop, but the Streamlit Cloud app kept failing. First it couldn't import `joblib` because the build was using a very new Python version that skipped some packages. After I got past that, the models refused to load with a version-mismatch error — I had trained them on a newer scikit-learn than the one Cloud installed. I fixed it by pinning scikit-learn in `requirements.txt` to the same version the models were trained on. I also learned that Streamlit Cloud ignores `runtime.txt`, so the Python version has to be chosen in the Advanced settings when deploying. Getting the CSV separator right (the UCI file uses `;`) and keeping the joblib filenames identical between the training script and the app were two smaller things that also tripped me up early on.

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
py model/train_models.py
py -m streamlit run app.py
```

## Streamlit app features

- CSV upload for test data
- Model selection dropdown
- All six evaluation metrics per model
- Confusion matrix heatmap and classification report
- Comparison of every model on the uploaded test set
- F1-score bar chart across models
- Downloadable predictions
