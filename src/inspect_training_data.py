
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent.parent
file_path = root / "data" / "raw" / "hai-train1.txt"

print("Loading training data...")
df = pd.read_csv(file_path)

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

print("\n--- Training dataset ---")
print("Records:", len(df))
print("Columns:", len(df.columns))
print("First timestamp:", df["timestamp"].min())
print("Last timestamp:", df["timestamp"].max())

print("\n--- Data quality ---")
print("Missing values:", df.isna().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print("Invalid timestamps:", df["timestamp"].isna().sum())

features = df.drop(columns=["timestamp"])
constant_cols = features.columns[
    features.nunique(dropna=False) <= 1
].tolist()

print("\n--- Feature checks ---")
print("Number of process features:", len(features.columns))
print("Constant features:", len(constant_cols))
print(constant_cols)

print("\n--- Memory usage ---")
print("DataFrame memory (MB):",
      round(df.memory_usage(deep=True).sum() / (1024**2), 2))
