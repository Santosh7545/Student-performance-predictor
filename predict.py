"""Run a sample inference with the saved model."""
from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = joblib.load(root / "models" / "student_pass_predictor.joblib")
    sample = pd.DataFrame(
        [[88, 16, 72, 7.5, 9]], columns=artifact["features"]
    )
    probability = artifact["model"].predict_proba(sample)[0, 1]
    decision = "Likely to pass" if probability >= 0.5 else "May need support"
    print(f"Prediction: {decision}")
    print(f"Pass probability: {probability:.1%}")


if __name__ == "__main__":
    main()
