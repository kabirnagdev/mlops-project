
from pathlib import Path

import joblib
import mlflow.sklearn
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_URI = "models:/mp033-canteen-champion@champion"

FEATURES = [
    "students_on_campus",
    "menu_price_rs",
    "temp_c",
    "exam_days_left",
    "special_dishes",
    "queue_minutes_yesterday",
    "day_type",
    "block",
]


def main():
    print("Loading registered champion model...")

    # Load the actual sklearn pipeline from MLflow
    model = mlflow.sklearn.load_model(MODEL_URI)

    # Export champion locally
    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "model.joblib"
    joblib.dump(model, model_path)

    print("Champion exported:", model_path)

    # Load it again to verify the saved artifact
    loaded_model = joblib.load(model_path)

    # Load five real feature records
    later_path = ROOT_DIR / "data" / "raw" / "later.csv"
    df = pd.read_csv(later_path)

    sample = df[FEATURES].head(5).copy()

    predictions = loaded_model.predict(sample)

    print("\nExport verification")
    print("-------------------")
    print("Model URI:", MODEL_URI)
    print("Input records:", len(sample))
    print("Predictions:", len(predictions))

    for index, prediction in enumerate(predictions, start=1):
        print(f"Record {index}: {prediction:.2f} plates")

    print("\nChampion model export successful.")


if __name__ == "__main__":
    main()
