"""Create a small, reproducible dataset for this educational demo."""
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(42)
    rows = 600
    attendance = rng.integers(45, 101, rows)
    study_hours = np.clip(rng.normal(14, 6, rows), 0, 35).round(1)
    previous_score = np.clip(rng.normal(62, 16, rows), 0, 100).round(1)
    sleep_hours = np.clip(rng.normal(7, 1.2, rows), 3, 10).round(1)
    assignments = rng.integers(0, 11, rows)

    # Latent score creates a realistic but intentionally synthetic target.
    score = (
        0.05 * attendance
        + 1.1 * study_hours
        + 0.42 * previous_score
        + 1.6 * sleep_hours
        + 2.0 * assignments
        + rng.normal(0, 10, rows)
    )
    passed = (score >= 61).astype(int)

    data = pd.DataFrame(
        {
            "attendance_percent": attendance,
            "weekly_study_hours": study_hours,
            "previous_exam_score": previous_score,
            "average_sleep_hours": sleep_hours,
            "assignments_completed": assignments,
            "passed": passed,
        }
    )
    destination = Path(__file__).resolve().parents[1] / "data" / "student_habits.csv"
    destination.parent.mkdir(exist_ok=True)
    data.to_csv(destination, index=False)
    print(f"Saved {len(data)} rows to {destination}")


if __name__ == "__main__":
    main()
