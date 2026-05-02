import pandas as pd
from pathlib import Path

path = Path("data/interim/team_game_logs_clean_2025-26_regular_season.parquet")
df = pd.read_parquet(path)

check = (
    df.groupby(["game_id", "home_away"])
      .size()
      .unstack(fill_value=0)
      .reset_index()
)

print(check[(check.get("HOME", 0) != 1) | (check.get("AWAY", 0) != 1)].head(20))
print("Problem games:", len(check[(check.get("HOME", 0) != 1) | (check.get("AWAY", 0) != 1)]))
