
"""
This module applies feature engineering to the cleaned Diabetes dataset.
It creates new features and saves the result ready for model training.
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "processed" / "train_v1.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "train_features_v1.csv"


def load_data(path: Path) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame."""
    return pd.read_csv(path)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering steps and return the enriched DataFrame.

    New features created:
    - bmi_age_interaction: interaction between BMI and age
    - bp_bmi_interaction: interaction between blood pressure and BMI
    - s1_s2_ratio: ratio between two serum measurements
    """
    df = df.copy()

    # Interaction between BMI and age
    df["bmi_age_interaction"] = df["bmi"] * df["age"]

    # Interaction between blood pressure and BMI
    df["bp_bmi_interaction"] = df["bp"] * df["bmi"]

    # Ratio of serum measurements (avoid division by zero)
    df["s1_s2_ratio"] = df["s1"] / (df["s2"] + 1e-9)

    return df


def save_data(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to CSV without the index.

    Raises FileExistsError if the output file already exists.
    """
    if path.exists():
        raise FileExistsError(f"Output file already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    """Run the full feature engineering pipeline."""
    print("Loading processed data...")
    df = load_data(INPUT_PATH)
    print(f"Rows before feature engineering: {len(df)}")

    print("\nApplying feature engineering...")
    df = feature_engineering(df)

    print("\nSaving dataset...")
    save_data(df, OUTPUT_PATH)

    print(f"Final shape: {df.shape}")
    print(f"\nOutput path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
