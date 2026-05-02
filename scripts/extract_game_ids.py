from pathlib import Path
import pandas as pd

INTERIM_DIR = Path("data/interim")
INTERIM_DIR.mkdir(parents=True, exist_ok=True)


def normalize_label(value: str) -> str:
    return value.lower().replace(" ", "_")


def extract_game_ids(season: str, season_type: str) -> None:
    raw_path = Path(
        f"data/raw/game_logs/player_game_logs_raw_{season}_{normalize_label(season_type)}.parquet"
    )
    out_path = INTERIM_DIR / f"game_ids_{season}_{normalize_label(season_type)}.parquet"

    df = pd.read_parquet(raw_path)
    df.columns = [c.lower() for c in df.columns]

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])

    game_ids = (
        df[["game_id", "game_date"]]
        .drop_duplicates(subset=["game_id"])
        .sort_values(["game_date", "game_id"])
        .reset_index(drop=True)
    )

    game_ids["game_id"] = game_ids["game_id"].astype(str)
    game_ids["season"] = season
    game_ids["season_type"] = season_type

    game_ids.to_parquet(out_path, index=False)

    print(f"Saved {len(game_ids)} unique game IDs to {out_path}")
    print(game_ids.head())


def main() -> None:
    season = "2025-26"
    for season_type in ["Regular Season", "Playoffs"]:
        extract_game_ids(season, season_type)


if __name__ == "__main__":
    main()