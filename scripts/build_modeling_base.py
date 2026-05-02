from __future__ import annotations

from pathlib import Path
import pandas as pd


PLAYER_LOGS_PATH = Path("data/raw/game_logs/player_game_logs_raw_2025-26_regular_season.parquet")
TEAM_CONTEXT_PATH = Path("data/interim/team_game_logs_clean_2025-26_regular_season.parquet")
OUT_PATH = Path("data/interim/modeling_base_2025-26_regular_season.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_player_logs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Player logs file not found: {path}")

    df = pd.read_parquet(path).copy()
    df.columns = [c.lower() for c in df.columns]

    df["game_id"] = df["game_id"].astype(str)
    df["player_id"] = df["player_id"].astype(str)
    df["team_id"] = df["team_id"].astype(str)

    # Normalize game date
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])
    elif "game_date_est" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date_est"])
    else:
        raise ValueError("No game_date column found in player logs")

    return df


def load_team_context(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Team context file not found: {path}")

    df = pd.read_parquet(path).copy()
    df.columns = [c.lower() for c in df.columns]

    df["game_id"] = df["game_id"].astype(str)
    df["team_id"] = df["team_id"].astype(str)

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])

    return df


def select_team_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    desired = [
        "game_id",
        "team_id",
        "team_abbreviation",
        "team_name",
        "matchup",
        "wl",
        "pts",
        "fgm",
        "fga",
        "fg_pct",
        "fg3m",
        "fg3a",
        "fg3_pct",
        "ftm",
        "fta",
        "ft_pct",
        "oreb",
        "dreb",
        "reb",
        "ast",
        "stl",
        "blk",
        "tov",
        "pf",
        "plus_minus",
        "season",
        "season_type",
        "home_away",
        "opponent_abbreviation",
    ]
    available = [c for c in desired if c in df.columns]
    out = df[available].copy()

    rename_map = {
        "team_abbreviation": "team_ctx_abbreviation",
        "team_name": "team_ctx_name",
        "matchup": "team_ctx_matchup",
        "wl": "team_ctx_wl",
        "pts": "team_ctx_pts",
        "fgm": "team_ctx_fgm",
        "fga": "team_ctx_fga",
        "fg_pct": "team_ctx_fg_pct",
        "fg3m": "team_ctx_fg3m",
        "fg3a": "team_ctx_fg3a",
        "fg3_pct": "team_ctx_fg3_pct",
        "ftm": "team_ctx_ftm",
        "fta": "team_ctx_fta",
        "ft_pct": "team_ctx_ft_pct",
        "oreb": "team_ctx_oreb",
        "dreb": "team_ctx_dreb",
        "reb": "team_ctx_reb",
        "ast": "team_ctx_ast",
        "stl": "team_ctx_stl",
        "blk": "team_ctx_blk",
        "tov": "team_ctx_tov",
        "pf": "team_ctx_pf",
        "plus_minus": "team_ctx_plus_minus",
        "season": "team_ctx_season",
        "season_type": "team_ctx_season_type",
        "home_away": "home_away",
        "opponent_abbreviation": "opponent_abbreviation",
    }
    return out.rename(columns=rename_map)


def build_modeling_base(player_df: pd.DataFrame, team_df: pd.DataFrame) -> pd.DataFrame:
    team_ctx = select_team_context_columns(team_df)

    merged = player_df.merge(
        team_ctx,
        on=["game_id", "team_id"],
        how="left",
        validate="many_to_one",
        suffixes=("_player", "_team"),
    )

    # Resolve duplicated game date columns after merge
    if "game_date_player" in merged.columns:
        merged["game_date"] = merged["game_date_player"]
    elif "game_date_x" in merged.columns:
        merged["game_date"] = merged["game_date_x"]
    elif "game_date" not in merged.columns:
        raise ValueError("No usable game_date column found after merge")

    merged = merged.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)
    return merged


def validate_modeling_base(df: pd.DataFrame) -> None:
    print("Modeling base shape:", df.shape)
    print("Unique games:", df["game_id"].nunique())
    print("Unique players:", df["player_id"].nunique())

    key_missing = df[["game_id", "player_id", "team_id"]].isna().sum()
    print("\nMissing key values:")
    print(key_missing)

    context_missing = {
        "home_away": df["home_away"].isna().sum() if "home_away" in df.columns else None,
        "opponent_abbreviation": df["opponent_abbreviation"].isna().sum() if "opponent_abbreviation" in df.columns else None,
        "team_ctx_pts": df["team_ctx_pts"].isna().sum() if "team_ctx_pts" in df.columns else None,
    }
    print("\nMissing joined context:")
    print(context_missing)

    if "home_away" in df.columns:
        print("\nHome/Away distribution:")
        print(df["home_away"].value_counts(dropna=False))


def main() -> None:
    print("Loading player logs...")
    player_df = load_player_logs(PLAYER_LOGS_PATH)

    print("Loading team context...")
    team_df = load_team_context(TEAM_CONTEXT_PATH)

    print("Building modeling base...")
    modeling_base = build_modeling_base(player_df, team_df)

    print("Validating...")
    validate_modeling_base(modeling_base)

    modeling_base.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved modeling base to: {OUT_PATH}")


if __name__ == "__main__":
    main()