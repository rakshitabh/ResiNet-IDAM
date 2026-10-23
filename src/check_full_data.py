
from pathlib import Path
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent
file_path = project_dir / "data" / "raw" / "hai-train1.txt"

print("Loading full dataset...")
df = pd.read_csv(file_path)

print("\n--- Full dataset ---")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))
print("File size (MB):", round(file_path.stat().st_size / (1024 * 1024), 2))

print("\n--- Timestamp continuity ---")
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

print("Invalid timestamps:", df["timestamp"].isna().sum())
print("First timestamp:", df["timestamp"].min())
print("Last timestamp:", df["timestamp"].max())

time_diff = df["timestamp"].diff().dropna()
print("\nMost common time intervals:")
print(time_diff.value_counts().head(5).to_string())

print("\n--- Constant columns ---")
constant_cols = [
    col for col in df.columns
    if df[col].nunique(dropna=False) <= 1
]
print(constant_cols if constant_cols else "None")

print("\n--- Data quality ---")
print("Missing values:", df.isna().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
