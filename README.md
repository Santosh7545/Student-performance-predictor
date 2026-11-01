# Student Performance Predictor

An end-to-end machine-learning project that predicts whether a student is likely to pass based on study habits and attendance. It demonstrates a practical classification workflow: data preparation, model training, evaluation, model persistence, and inference.

> **Note:** The included dataset is synthetically generated for demonstration and learning; it must not be used for real academic decisions.

## What it does

- Generates a reproducible student-habits dataset
- Trains and compares Logistic Regression and Random Forest classifiers
- Selects the model with the better F1 score
- Saves the trained pipeline to `models/student_pass_predictor.joblib`
- Predicts pass probability for a new student from the command line

## Tech stack

Python, pandas, scikit-learn, joblib

## Project structure

```
student-performance-predictor/
├── data/                 # Generated sample dataset
├── models/               # Saved model (created after training)
├── src/
│   ├── generate_data.py
│   ├── train.py
│   └── predict.py
├── requirements.txt
└── README.md
```

## Run locally

```bash
git clone <your-repository-url>
cd student-performance-predictor
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt

python src/generate_data.py
python src/train.py
python src/predict.py
```

## Sample result

The training script reports accuracy, precision, recall and F1 score on a held-out test split, then saves the strongest model. The generated data is deliberately small so that the full workflow runs quickly.

## Talking points for an interview / CRT

1. I framed the problem as binary classification: pass or not pass.
2. I used a pipeline so preprocessing and modelling remain consistent during inference.
3. I evaluated F1, not only accuracy, because both false positives and false negatives matter.
4. This is a learning demo built with synthetic data; a real deployment needs representative data, fairness checks and human oversight.

## Future improvements

- Add a Streamlit dashboard for interactive predictions.
- Track experiments with MLflow.
- Validate fairness across relevant groups using responsibly collected data.

