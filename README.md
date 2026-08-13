# Student Outcome Classifier

M.Tech (AIML) - Machine Learning - Assignment 2
**Name:** SUDHANVA S A  |  **BITS ID:** 2025AC05568

- **Live app:** https://student-outcome-classifier-gyeofgb4xk59585mm4pvhm.streamlit.app/
- **GitHub:** https://github.com/sudhanva2025AC05568/student-outcome-classifier

> **To test the app:** open the live link, upload the `test_data.csv` file from this repository, and select `Target` as the outcome column. If the app shows "app is sleeping," click "Yes, get this app back up" and wait a few seconds.

---

## a. Problem statement

In this project I try to predict what happens to a student by the end of their course - whether they will **drop out**, stay **enrolled**, or **graduate**. Since there are three possible outcomes, this is a multi-class classification problem.

I chose this problem because dropout is a real issue for colleges. If we can predict early which students are likely to leave, the college can try to help them in time.

## b. Dataset description

| Item | Detail |
|---|---|
| Source | UCI Machine Learning Repository, dataset ID 697 |
| Link | https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success |
| Number of rows | 4,424 |
| Number of features | 36 |
| Target column | `Target` (Dropout / Enrolled / Graduate) |
| Type of problem | Multi-class classification (3 classes) |
| Class balance | Graduate around 50%, Dropout around 32%, Enrolled around 18% |
| Missing values | None, but the code still handles them just in case |
| Feature types | Some are numbers (like grades and age) and some are categories (like course and nationality) |

The dataset includes information about students’ background, course, and marks from the first two semesters, which are used as input features for prediction.

## c. GitHub repository link

https://github.com/sudhanva2025AC05568/student-outcome-classifier

The repository has the code (`app.py`, `train_models.py`), `requirements.txt`, `README.md`, `test_data.csv`, and the `model` folder with the saved models.

## d. Models used

I trained all six models on the same data. I split the data into 80% for training and 20% for testing, and used the same split for every model so the comparison is fair. Before training, I filled any missing values, scaled the numeric columns, and one-hot encoded the category columns.

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

| ML Model Name | Observation |
|---|---|
| Logistic Regression | Worked really well even though it is a simple model. It came second in F1 and MCC. |
| Decision Tree | It was average. A single tree tends to overfit, so it did worse than the forest-based models. |
| kNN | One of the weakest. After one-hot encoding there are many columns, and kNN does not work well when there are too many columns. |
| Naive Bayes | The weakest one. It assumes all features are independent, but here they are not (for example, grades and units passed are related). |
| Random Forest | The best model. It combines many trees, so it does not overfit the way a single tree does. |
| Gradient Boosting | Almost as good as Random Forest and had the best AUC, but its F1 was slightly lower. |
| Overall winner | **Random Forest** - it had the best F1 and MCC. I chose based on MCC because the classes are imbalanced, and MCC handles imbalance better than accuracy. |

All the models did better than simply guessing the biggest class (which would give around 50% accuracy). Every model was good at predicting Graduate and Dropout, but all of them found the Enrolled class hard. This makes sense, because an enrolled student is still studying and could still drop out or graduate later.

## Challenges I faced

The hardest part was deploying the app. It worked fine on my laptop but kept failing on Streamlit Cloud. First it could not find `joblib`, and then the models would not load because Streamlit was using a different scikit-learn version than the one I trained with. I fixed it by setting the same scikit-learn version in `requirements.txt`. I also learned that Streamlit ignores `runtime.txt`, so the Python version has to be chosen in the Advanced settings while deploying. Two smaller things that confused me early on were the CSV separator (the UCI file uses `;` instead of `,`) and making sure the saved model file names matched exactly in both the training code and the app.

## How to run it

```bash
pip install -r requirements.txt
py model/train_models.py
py -m streamlit run app.py
```

## What the app can do

- Upload a test CSV file
- Pick a model from a dropdown
- See all six metrics for the selected model
- See the confusion matrix and classification report
- Compare all models side by side
- See an F1 score bar chart
