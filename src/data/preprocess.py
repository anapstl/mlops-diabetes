
"""
This module cleans the raw Diabetes dataset by handling missing values,
removing duplicates, and fixing data types.
It outputs a processed dataset ready for machine learning.
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_PATH = BASE_DIR / "data" / "raw" / "diabetes_dataset.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "train_v1.csv"


def load_data(path: Path) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame."""
    return pd.read_csv(path)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows containing missing values."""
    return df.dropna()


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the dataset."""
    return df.drop_duplicates()


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all columns to numeric types."""
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps and return the cleaned DataFrame."""
    df = df.copy()
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = fix_data_types(df)
    return df


def save_data(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to CSV without the index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    """Run the full preprocessing pipeline."""
    print("Loading raw data...")
    df = load_data(RAW_PATH)
    rows_before = len(df)
    print(f"Rows before cleaning: {rows_before}")

    print("Cleaning data...")
    df = clean_data(df)
    rows_after = len(df)
    print(f"Rows after cleaning: {rows_after}")

    print("Saving processed data...")
    save_data(df, PROCESSED_PATH)

    print(f"Final shape: {df.shape}")
    print(f"Output path: {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
