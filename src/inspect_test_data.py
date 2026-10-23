
from pathlib import Path
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent
file_path = project_dir / "data" / "raw" / "hai-test1.txt"

print("Loading first 1,000 test records...")
df = pd.read_csv(file_path, nrows=1000)

print("\n--- Dataset information ---")
print("Rows inspected:", len(df))
print("Number of columns:", len(df.columns))

print("\n--- Column names ---")
print(df.columns.tolist())

print("\n--- Data types ---")
print(df.dtypes.to_string())

print("\n--- Missing values ---")
print(df.isna().sum().sum())

print("\n--- First 5 rows ---")
print(df.head().to_string())

label_candidates = [
    col for col in df.columns
    if any(word in col.lower() for word in
           ["attack", "label", "class", "target", "anomaly"])
]

print("\n--- Possible label columns ---")
print(label_candidates if label_candidates else "None found by name")
