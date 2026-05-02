from __future__ import annotations

from pathlib import Path
import json
import math
import pandas as pd


FEATURES_PATH = Path("data/processed/features_next_game_points_2025-26_regular_season.parquet")
PBP_DIR = Path("data/raw/play_by_play/2025-26/regular_season")
OUT_PATH = Path("data/processed/event_sequences_next_game_points_2025-26_regular_season.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MAX_EVENTS = 256  # first-pass cap for manageable sequence size


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")

    df = pd.read_parquet(path).copy()
    df.columns = [c.lower() for c in df.columns]

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])
    if "target_next_game_date" in df.columns:
        df["target_next_game_date"] = pd.to_datetime(df["target_next_game_date"])

    df["player_id"] = df["player_id"].astype(str)
    df["team_id"] = df["team_id"].astype(str)
    df["game_id"] = df["game_id"].astype(str)
    if "target_next_game_id" in df.columns:
        df["target_next_game_id"] = df["target_next_game_id"].astype(str)

    return df


def add_previous_game_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)
    out["previous_game_id"] = out.groupby("player_id")["game_id"].shift(1)
    out["previous_game_date"] = out.groupby("player_id")["game_date"].shift(1)
    return out


def load_play_by_play(game_id: str) -> pd.DataFrame:
    path = PBP_DIR / f"{game_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"PBP file missing for game_id={game_id}: {path}")

    df = pd.read_parquet(path).copy()

    # Preserve original case if present, but standardize access with a lowercase map
    col_map = {c.lower(): c for c in df.columns}

    def get_col(name: str) -> str | None:
        return col_map.get(name.lower())

    # Required-ish fields
    action_number_col = get_col("actionNumber") or get_col("action_number")
    period_col = get_col("period")
    clock_col = get_col("clock")
    action_type_col = get_col("actionType") or get_col("action_type")
    subtype_col = get_col("subType") or get_col("sub_type")
    team_id_col = get_col("teamId") or get_col("team_id")
    person_id_col = get_col("personId") or get_col("person_id")
    points_total_col = get_col("pointsTotal") or get_col("points_total")
    score_home_col = get_col("scoreHome") or get_col("score_home")
    score_away_col = get_col("scoreAway") or get_col("score_away")
    description_col = get_col("description")

    keep = [
        c
        for c in [
            action_number_col,
            period_col,
            clock_col,
            action_type_col,
            subtype_col,
            team_id_col,
            person_id_col,
            points_total_col,
            score_home_col,
            score_away_col,
            description_col,
        ]
        if c is not None
    ]

    out = df[keep].copy()

    rename = {}
    if action_number_col:
        rename[action_number_col] = "action_number"
    if period_col:
        rename[period_col] = "period"
    if clock_col:
        rename[clock_col] = "clock"
    if action_type_col:
        rename[action_type_col] = "action_type"
    if subtype_col:
        rename[subtype_col] = "subtype"
    if team_id_col:
        rename[team_id_col] = "team_id"
    if person_id_col:
        rename[person_id_col] = "person_id"
    if points_total_col:
        rename[points_total_col] = "points_total"
    if score_home_col:
        rename[score_home_col] = "score_home"
    if score_away_col:
        rename[score_away_col] = "score_away"
    if description_col:
        rename[description_col] = "description"

    out = out.rename(columns=rename)

    if "action_number" in out.columns:
        out = out.sort_values("action_number").reset_index(drop=True)

    if "team_id" in out.columns:
        out["team_id"] = out["team_id"].astype(str)
    if "person_id" in out.columns:
        out["person_id"] = out["person_id"].astype(str)

    return out


def safe_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, str) and value.lower() == "nan":
        return None
    if isinstance(value, pd.Timestamp):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def build_event_token_dict(row: pd.Series) -> dict:
    return {
        "period": safe_value(row["period"]) if "period" in row else None,
        "clock": safe_value(row["clock"]) if "clock" in row else None,
        "action_type": safe_value(row["action_type"]) if "action_type" in row else None,
        "subtype": safe_value(row["subtype"]) if "subtype" in row else None,
        "team_id": safe_value(row["team_id"]) if "team_id" in row else None,
        "person_id": safe_value(row["person_id"]) if "person_id" in row else None,
        "points_total": safe_value(row["points_total"]) if "points_total" in row else None,
        "score_home": safe_value(row["score_home"]) if "score_home" in row else None,
        "score_away": safe_value(row["score_away"]) if "score_away" in row else None,
        "description": safe_value(row["description"]) if "description" in row else None,
    }


def build_sequence_for_game(game_id: str, max_events: int = MAX_EVENTS) -> list[dict]:
    pbp_df = load_play_by_play(game_id)

    tokens = [build_event_token_dict(row) for _, row in pbp_df.iterrows()]
    if max_events is not None:
        tokens = tokens[:max_events]

    return tokens


def build_event_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = add_previous_game_id(df)

    # We need a previous game to provide sequence input
    out = out.dropna(subset=["previous_game_id"]).reset_index(drop=True)
    out["previous_game_id"] = out["previous_game_id"].astype(str)

    unique_prev_games = out["previous_game_id"].drop_duplicates().tolist()
    print(f"Unique previous games to load: {len(unique_prev_games)}")

    sequence_cache: dict[str, str] = {}
    failed_games: list[str] = []

    for i, game_id in enumerate(unique_prev_games, start=1):
        try:
            tokens = build_sequence_for_game(game_id)
            sequence_cache[game_id] = json.dumps(tokens, ensure_ascii=False)
            if i % 100 == 0 or i == len(unique_prev_games):
                print(f"[{i}/{len(unique_prev_games)}] cached sequence for game_id={game_id}")
        except Exception as exc:
            failed_games.append(game_id)
            sequence_cache[game_id] = json.dumps([], ensure_ascii=False)
            print(f"[{i}/{len(unique_prev_games)}] FAILED sequence for game_id={game_id}: {exc}")

    out["event_sequence_json"] = out["previous_game_id"].map(sequence_cache)

    if failed_games:
        failed_path = OUT_PATH.parent / "failed_event_sequence_games.txt"
        failed_path.write_text("\n".join(failed_games), encoding="utf-8")
        print(f"Failed sequence games saved to: {failed_path}")

    final_cols = [
        "player_id",
        "player_name",
        "team_id",
        "team_abbreviation",
        "game_id",
        "game_date",
        "previous_game_id",
        "previous_game_date",
        "home_away",
        "opponent_abbreviation",
        "pts_roll3",
        "pts_roll5",
        "reb_roll3",
        "reb_roll5",
        "ast_roll3",
        "ast_roll5",
        "min_roll3",
        "min_roll5",
        "team_ctx_pace_proxy",
        "target_next_game_pts",
        "target_next_game_date",
        "target_next_game_id",
        "event_sequence_json",
    ]
    final_cols = [c for c in final_cols if c in out.columns]

    out = out[final_cols].copy()
    return out


def validate_event_dataset(df: pd.DataFrame) -> None:
    print("Event dataset shape:", df.shape)
    print("Unique players:", df["player_id"].nunique())
    print("Unique games:", df["game_id"].nunique())

    empty_sequences = (df["event_sequence_json"] == "[]").sum()
    print("Empty sequences:", int(empty_sequences))

    print("\nPreview:")
    print(df.head())

    if len(df) > 0:
        first_seq = json.loads(df.iloc[0]["event_sequence_json"])
        print("\nFirst sequence length:", len(first_seq))
        if first_seq:
            print("First event token example:")
            print(first_seq[0])


def main() -> None:
    print("Loading feature dataset...")
    df = load_features(FEATURES_PATH)

    print("Building event-sequence dataset...")
    event_df = build_event_dataset(df)

    print("Validating...")
    validate_event_dataset(event_df)

    event_df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved event sequence dataset to: {OUT_PATH}")


if __name__ == "__main__":
    main()