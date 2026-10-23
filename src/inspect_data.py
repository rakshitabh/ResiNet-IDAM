
from pathlib import Path
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent
file_path = project_dir / "data" / "raw" / "hai-train1.txt"

print("Loading first 1,000 rows...")

df = pd.read_csv(file_path, nrows=1000)

print("\n--- Dataset information ---")
print("Rows inspected:", len(df))
print("Number of columns:", len(df.columns))

print("\n--- Column names ---")
print(df.columns.tolist())

print("\n--- Data types ---")
print(df.dtypes.to_string())

print("\n--- Missing values ---")
print(df.isnull().sum().sum())

print("\n--- Duplicate rows ---")
print(df.duplicated().sum())

print("\n--- Timestamp check ---")
df["timestamp"] = pd.to_datetime(
    df["timestamp"], errors="coerce"
)
print("Invalid timestamps:", df["timestamp"].isnull().sum())
print("First timestamp:", df["timestamp"].min())
print("Last timestamp:", df["timestamp"].max())

print("\n--- Sample statistics ---")
print(df.describe().T.head(10).to_string())
