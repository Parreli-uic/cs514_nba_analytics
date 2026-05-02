from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


INPUT_PATH = Path("data/interim/modeling_base_2025-26_regular_season.parquet")
OUT_PATH = Path("data/processed/features_next_game_points_2025-26_regular_season.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


ROLLING_WINDOWS = [3, 5]


def load_modeling_base(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Modeling base file not found: {path}")

    df = pd.read_parquet(path).copy()
    df.columns = [c.lower() for c in df.columns]

    df["game_id"] = df["game_id"].astype(str)
    df["player_id"] = df["player_id"].astype(str)
    df["team_id"] = df["team_id"].astype(str)

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])

    df = df.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)
    return df


def add_basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    numeric_defaults = [
        "min",
        "pts",
        "reb",
        "ast",
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
        "stl",
        "blk",
        "tov",
        "pf",
        "plus_minus",
        "team_ctx_pts",
        "team_ctx_fga",
        "team_ctx_fgm",
        "team_ctx_fg_pct",
        "team_ctx_fg3a",
        "team_ctx_fg3m",
        "team_ctx_fg3_pct",
        "team_ctx_fta",
        "team_ctx_ftm",
        "team_ctx_ft_pct",
        "team_ctx_reb",
        "team_ctx_ast",
        "team_ctx_tov",
        "team_ctx_plus_minus",
    ]
    for col in numeric_defaults:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "wl" in out.columns:
        out["wl_flag"] = (out["wl"] == "W").astype(int)
    elif "team_ctx_wl" in out.columns:
        out["wl_flag"] = (out["team_ctx_wl"] == "W").astype(int)
    else:
        out["wl_flag"] = np.nan

    if "home_away" in out.columns:
        out["is_home"] = (out["home_away"] == "HOME").astype(int)
    else:
        out["is_home"] = np.nan

    return out


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    base_stats = ["pts", "reb", "ast", "min", "fga", "fg3a", "fta", "tov", "plus_minus"]
    for stat in base_stats:
        if stat not in out.columns:
            continue

        grouped = out.groupby("player_id")[stat]
        prev = grouped.shift(1)
        out[f"{stat}_prev"] = prev

        for window in ROLLING_WINDOWS:
            out[f"{stat}_roll{window}"] = (
                grouped.shift(1).rolling(window=window, min_periods=1).mean()
            )

    return out


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if {"team_ctx_fga", "team_ctx_fta", "team_ctx_tov", "team_ctx_reb"}.issubset(out.columns):
        out["team_ctx_pace_proxy"] = (
            out["team_ctx_fga"].fillna(0)
            + 0.44 * out["team_ctx_fta"].fillna(0)
            + out["team_ctx_tov"].fillna(0)
            - out["team_ctx_reb"].fillna(0)
        )

    if "min" in out.columns and "pts" in out.columns:
        out["pts_per_min"] = np.where(out["min"] > 0, out["pts"] / out["min"], np.nan)

    if "min" in out.columns and "ast" in out.columns:
        out["ast_per_min"] = np.where(out["min"] > 0, out["ast"] / out["min"], np.nan)

    if "min" in out.columns and "reb" in out.columns:
        out["reb_per_min"] = np.where(out["min"] > 0, out["reb"] / out["min"], np.nan)

    return out


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target_next_game_pts"] = out.groupby("player_id")["pts"].shift(-1)
    out["target_next_game_date"] = out.groupby("player_id")["game_date"].shift(-1)
    out["target_next_game_id"] = out.groupby("player_id")["game_id"].shift(-1)
    return out


def select_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "season_id",
        "player_id",
        "player_name",
        "team_id",
        "team_abbreviation",
        "game_id",
        "game_date",
        "matchup",
        "wl",
        "home_away",
        "is_home",
        "opponent_abbreviation",
        "wl_flag",
        "pts",
        "reb",
        "ast",
        "min",
        "fga",
        "fg3a",
        "fta",
        "tov",
        "plus_minus",
        "pts_prev",
        "reb_prev",
        "ast_prev",
        "min_prev",
        "pts_roll3",
        "pts_roll5",
        "reb_roll3",
        "reb_roll5",
        "ast_roll3",
        "ast_roll5",
        "min_roll3",
        "min_roll5",
        "fga_roll3",
        "fga_roll5",
        "fg3a_roll3",
        "fg3a_roll5",
        "fta_roll3",
        "fta_roll5",
        "tov_roll3",
        "tov_roll5",
        "plus_minus_roll3",
        "plus_minus_roll5",
        "pts_per_min",
        "ast_per_min",
        "reb_per_min",
        "team_ctx_pts",
        "team_ctx_fga",
        "team_ctx_fg_pct",
        "team_ctx_fg3a",
        "team_ctx_fg3_pct",
        "team_ctx_fta",
        "team_ctx_ft_pct",
        "team_ctx_reb",
        "team_ctx_ast",
        "team_ctx_tov",
        "team_ctx_plus_minus",
        "team_ctx_pace_proxy",
        "target_next_game_pts",
        "target_next_game_date",
        "target_next_game_id",
    ]
    existing = [c for c in keep if c in df.columns]
    out = df[existing].copy()

    out = out.dropna(subset=["target_next_game_pts"]).reset_index(drop=True)
    return out


def main() -> None:
    print("Loading modeling base...")
    df = load_modeling_base(INPUT_PATH)

    print("Adding cleaned fields...")
    df = add_basic_cleaning(df)

    print("Adding rolling features...")
    df = add_rolling_features(df)

    print("Adding context features...")
    df = add_context_features(df)

    print("Adding next-game target...")
    df = add_target(df)

    print("Selecting final feature columns...")
    features_df = select_model_columns(df)

    features_df.to_parquet(OUT_PATH, index=False)

    print(f"Saved feature dataset to: {OUT_PATH}")
    print("Shape:", features_df.shape)
    print("Unique players:", features_df["player_id"].nunique())
    print("Unique games:", features_df["game_id"].nunique())
    print("\nPreview:")
    print(features_df.head())


if __name__ == "__main__":
    main()