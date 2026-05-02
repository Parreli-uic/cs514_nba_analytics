from pathlib import Path
import pandas as pd

path = Path("data/interim/team_game_logs_clean_2025-26_regular_season.parquet")

df = pd.read_parquet(path)

print("Shape:", df.shape)
print("Unique games:", df["game_id"].nunique())
print("Unique teams:", df["team_id"].nunique())
print("\nHome/Away counts:")
print(df["home_away"].value_counts(dropna=False))
print("\nPreview:")
print(df.head())