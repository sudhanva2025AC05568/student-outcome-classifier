# Student Outcome Classifier

M.Tech (AIML) - Machine Learning - Assignment 2
**Name:** SUDHANVA S A  |  **BITS ID:** 2025AC05568

- **Live app:** https://student-outcome-classifier-gyeofgb4xk59585mm4pvhm.streamlit.app/
- **GitHub:** https://github.com/sudhanva2025AC05568/student-outcome-classifier

---

## a. Problem statement

In this project I try to predict what happens to a student at the end - whether they will
**drop out**, stay **enrolled**, or **graduate**. Since there are three possible outcomes, this
is a multi-class classification problem.

I chose this problem because dropout is a real issue for colleges. If we can predict early
which students might leave, the college can try to help them in time.

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

The data has information about the student's background, their course, and their marks in the
first two semesters. The semester marks turned out to be the most useful for prediction.

## c. GitHub repository link

https://github.com/sudhanva2025AC05568/student-outcome-classifier

The repo has the code (`app.py`, `train_models.py`), `requirements.txt`, `README.md`,
`test_data.csv`, and the `model` folder with the saved models.

## d. Models used

I trained all six models on the same data. I split the data into 80% for training and 20% for
testing, and kept the same split for every model so the comparison is fair. Before training, I
filled missing values, scaled the number columns, and one-hot encoded the category columns.

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
| Decision Tree | It was average. A single tree tends to overfit, so it did worse than the forest models. |
| kNN | One of the weakest. After one-hot encoding there are too many columns, and kNN does not work well when there are many columns. |
| Naive Bayes | The weakest one. It assumes all features are independent, but here they are not (like grades and units passed are related). |
| Random Forest | The best model. It uses many trees together, so it does not overfit like a single tree. |
| Gradient Boosting | Almost as good as Random Forest and had the best AUC, but its F1 was a little lower. |
| Overall winner | **Random Forest** - it had the best F1 and MCC. I chose based on MCC because the classes are imbalanced. |

All the models did better than just guessing the biggest class (which would give around 50%).
Every model was good at predicting Graduate and Dropout, but all of them found the Enrolled
class hard. This makes sense because an enrolled student is still studying and could still
drop out or graduate later.

## Challenges I faced

The hardest part was deploying the app. It worked fine on my laptop but kept failing on
Streamlit Cloud. First it could not find `joblib`, and then the models would not load because
Streamlit was using a different scikit-learn version than the one I trained with. I fixed it by
setting the same scikit-learn version in `requirements.txt`. I also learned that Streamlit
ignores `runtime.txt`, so the Python version has to be chosen in the Advanced settings while
deploying. Two smaller things that confused me early were the CSV separator (the UCI file uses
`;` instead of `,`) and making sure the saved model file names matched exactly in both the
training code and the app.

## How to run it

```bash
pip install -r requirements.txt
py model/train_models.py
py -m streamlit run app.py
```

## What the app can do

- Upload a test CSV file
- Pick a model from a dropdown
- See all six metrics for that model
- See the confusion matrix and classification report
- Compare all models together
- See an F1 score bar chart
