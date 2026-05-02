from __future__ import annotations

from pathlib import Path
import pandas as pd

from nba_api.stats.endpoints import leaguegamelog


RAW_DIR = Path("data/raw/team_context")
INTERIM_DIR = Path("data/interim")
RAW_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

def derive_home_away_by_game(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["home_away"] = "UNKNOWN"

    for game_id, group in df.groupby("game_id"):
        idx = group.index.tolist()

        if len(idx) != 2:
            continue

        row1 = group.iloc[0]
        row2 = group.iloc[1]

        m1 = str(row1["matchup"])
        m2 = str(row2["matchup"])

        if ("vs." in m1 and "@" in m2):
            df.loc[idx[0], "home_away"] = "HOME"
            df.loc[idx[1], "home_away"] = "AWAY"
            continue

        if ("@" in m1 and "vs." in m2):
            df.loc[idx[0], "home_away"] = "AWAY"
            df.loc[idx[1], "home_away"] = "HOME"
            continue

        if ("@" in m1 and "@" in m2):
            away1 = m1.split("@")[0].strip()
            away2 = m2.split("@")[0].strip()

            if away1 == row1["team_abbreviation"] and away2 == row2["team_abbreviation"]:
                df.loc[idx[0], "home_away"] = "AWAY"
                df.loc[idx[1], "home_away"] = "HOME"
                continue

            if away1 == row2["team_abbreviation"] and away2 == row1["team_abbreviation"]:
                df.loc[idx[0], "home_away"] = "HOME"
                df.loc[idx[1], "home_away"] = "AWAY"
                continue

        if ("vs." in m1 and "vs." in m2):
            home1 = m1.split("vs.")[0].strip()
            home2 = m2.split("vs.")[0].strip()

            if home1 == row1["team_abbreviation"] and home2 == row2["team_abbreviation"]:
                df.loc[idx[0], "home_away"] = "HOME"
                df.loc[idx[1], "home_away"] = "AWAY"
                continue

            if home1 == row2["team_abbreviation"] and home2 == row1["team_abbreviation"]:
                df.loc[idx[0], "home_away"] = "AWAY"
                df.loc[idx[1], "home_away"] = "HOME"
                continue

        df.loc[idx[0], "home_away"] = "HOME"
        df.loc[idx[1], "home_away"] = "AWAY"

    return df

def normalize_label(value: str) -> str:
    return value.lower().replace(" ", "_")


def fetch_team_game_logs(
    season: str = "2025-26",
    season_type: str = "Regular Season",
) -> pd.DataFrame:
    response = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        player_or_team_abbreviation="T",
    )
    return response.get_data_frames()[0]


def save_raw_team_logs(df: pd.DataFrame, season: str, season_type: str) -> Path:
    outpath = RAW_DIR / f"team_game_logs_raw_{season}_{normalize_label(season_type)}.parquet"
    df.to_parquet(outpath, index=False)
    return outpath


def parse_home_away(matchup: str) -> str:
    if "vs." in matchup:
        return "HOME"
    if "@" in matchup:
        return "AWAY"
    return "UNKNOWN"


def parse_opponent(matchup: str, team_abbreviation: str) -> str:
    if not isinstance(matchup, str):
        return ""
    parts = matchup.replace("vs.", "@").split("@")
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) != 2:
        return ""
    if parts[0] == team_abbreviation:
        return parts[1]
    if parts[1] == team_abbreviation:
        return parts[0]
    return ""


def clean_team_game_logs(
    df: pd.DataFrame,
    season: str,
    season_type: str,
) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [c.lower() for c in cleaned.columns]

    cleaned["game_id"] = cleaned["game_id"].astype(str)
    cleaned["team_id"] = cleaned["team_id"].astype(str)

    if "game_date" in cleaned.columns:
        cleaned["game_date"] = pd.to_datetime(cleaned["game_date"])

    cleaned["season"] = season
    cleaned["season_type"] = season_type

    cleaned = derive_home_away_by_game(cleaned)
    cleaned["opponent_abbreviation"] = cleaned.apply(
        lambda row: parse_opponent(row["matchup"], row["team_abbreviation"]),
        axis=1,
    )

    cleaned = cleaned.drop_duplicates()

    return cleaned


def save_clean_team_logs(df: pd.DataFrame, season: str, season_type: str) -> Path:
    outpath = INTERIM_DIR / f"team_game_logs_clean_{season}_{normalize_label(season_type)}.parquet"
    df.to_parquet(outpath, index=False)
    return outpath


def main() -> None:
    season = "2025-26"

    for season_type in ["Regular Season", "Playoffs"]:
        print(f"Fetching team game logs for {season} | {season_type}...")
        raw_df = fetch_team_game_logs(season=season, season_type=season_type)

        raw_path = save_raw_team_logs(raw_df, season, season_type)
        print(f"Saved raw team context to {raw_path}")

        clean_df = clean_team_game_logs(raw_df, season, season_type)
        clean_path = save_clean_team_logs(clean_df, season, season_type)
        print(f"Saved clean team context to {clean_path}")

        print("\nPreview:")
        print(
            clean_df[
                [
                    "game_id",
                    "game_date",
                    "team_id",
                    "team_abbreviation",
                    "matchup",
                    "home_away",
                    "opponent_abbreviation",
                    "wl",
                    "pts",
                ]
            ].head()
        )

        print("\nShape:", clean_df.shape)
        print("Unique games:", clean_df["game_id"].nunique())
        print("Unique teams:", clean_df["team_id"].nunique())
        print("-" * 80)


if __name__ == "__main__":
    main()