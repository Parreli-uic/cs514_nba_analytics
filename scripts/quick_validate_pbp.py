from pathlib import Path
import pandas as pd

pbp_dir = Path("data/raw/play_by_play/2025-26/regular_season")
files = list(pbp_dir.glob("*.parquet"))

print("Files:", len(files))

if not files:
    print("No files found.")
else:
    df = pd.read_parquet(files[0])
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nRow count:", len(df))