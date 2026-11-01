"""Train, compare, and save student pass-prediction models."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "attendance_percent",
    "weekly_study_hours",
    "previous_exam_score",
    "average_sleep_hours",
    "assignments_completed",
]


def evaluate(name, model, x_test, y_test):
    prediction = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, prediction),
        "precision": precision_score(y_test, prediction, zero_division=0),
        "recall": recall_score(y_test, prediction, zero_division=0),
        "f1": f1_score(y_test, prediction, zero_division=0),
    }
    print(f"\n{name}")
    for metric, value in metrics.items():
        print(f"{metric:>10}: {value:.3f}")
    return metrics["f1"]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = pd.read_csv(root / "data" / "student_habits.csv")
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES], data["passed"], test_size=0.2, random_state=42, stratify=data["passed"]
    )

    candidates = {
        "Logistic Regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=42))]
        ),
        "Random Forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=3, random_state=42),
    }

    scores = {}
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        scores[name] = evaluate(name, model, x_test, y_test)

    best_name = max(scores, key=scores.get)
    best_model = candidates[best_name]
    model_path = root / "models" / "student_pass_predictor.joblib"
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump({"model": best_model, "features": FEATURES}, model_path)
    print(f"\nSaved {best_name} to {model_path}")


if __name__ == "__main__":
    main()
