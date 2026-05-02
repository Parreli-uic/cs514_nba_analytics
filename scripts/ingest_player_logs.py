from pathlib import Path
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

RAW_DIR = Path("data/raw/game_logs")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def normalize_label(value: str) -> str:
    return value.lower().replace(" ", "_")


def fetch_player_game_logs(
    season: str = "2025-26",
    season_type: str = "Regular Season",
) -> pd.DataFrame:
    response = leaguegamelog.LeagueGameLog(
        season=season,
        player_or_team_abbreviation="P",
        season_type_all_star=season_type,
    )
    return response.get_data_frames()[0]


def save_raw_game_logs(df: pd.DataFrame, season: str, season_type: str) -> Path:
    outpath = RAW_DIR / (
        f"player_game_logs_raw_{season}_{normalize_label(season_type)}.parquet"
    )
    df.to_parquet(outpath, index=False)
    return outpath


def basic_clean_game_logs(
    df: pd.DataFrame,
    season: str,
    season_type: str,
) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [c.lower() for c in cleaned.columns]

    cleaned["game_id"] = cleaned["game_id"].astype(str)
    cleaned["player_id"] = cleaned["player_id"].astype(str)
    cleaned["team_id"] = cleaned["team_id"].astype(str)

    if "game_date" in cleaned.columns:
        cleaned["game_date"] = pd.to_datetime(cleaned["game_date"])

    cleaned["season"] = season
    cleaned["season_type"] = season_type

    cleaned = cleaned.drop_duplicates()
    return cleaned


def main() -> None:
    season = "2025-26"

    for season_type in ["Regular Season", "Playoffs"]:
        print(f"Fetching player game logs for {season} | {season_type}...")
        raw_df = fetch_player_game_logs(season=season, season_type=season_type)
        raw_path = save_raw_game_logs(raw_df, season, season_type)
        print(f"Saved raw logs to {raw_path}")

        clean_df = basic_clean_game_logs(raw_df, season, season_type)

        print("\nPreview:")
        print(clean_df.head())

        print("\nShape:", clean_df.shape)
        print("Unique games:", clean_df["game_id"].nunique())
        print("Unique players:", clean_df["player_id"].nunique())
        print("-" * 80)


if __name__ == "__main__":
    main()