from __future__ import annotations

from pathlib import Path
import json
import math
import pandas as pd


FEATURES_PATH = Path("data/processed/features_next_game_points_2025-26_regular_season.parquet")
PBP_DIR = Path("data/raw/play_by_play/2025-26/regular_season")
OUT_PATH = Path("data/processed/possession_sequences_next_game_points_2025-26_regular_season.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MAX_POSSESSIONS = 128


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


def parse_clock_to_seconds(clock_str: str | None) -> float:
    if clock_str is None or not isinstance(clock_str, str):
        return 0.0

    try:
        if not clock_str.startswith("PT"):
            return 0.0

        body = clock_str[2:]
        minutes = 0.0
        seconds = 0.0

        if "M" in body:
            minute_part, rest = body.split("M", 1)
            minutes = float(minute_part) if minute_part else 0.0
        else:
            rest = body

        if rest.endswith("S"):
            rest = rest[:-1]
        seconds = float(rest) if rest else 0.0

        return minutes * 60.0 + seconds
    except Exception:
        return 0.0


def safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.lower() == "nan":
            return default
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def safe_str(value, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str) and value.lower() == "nan":
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    return str(value)


def load_play_by_play(game_id: str) -> pd.DataFrame:
    path = PBP_DIR / f"{game_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"PBP file missing for game_id={game_id}: {path}")

    df = pd.read_parquet(path).copy()
    col_map = {c.lower(): c for c in df.columns}

    def get_col(name: str) -> str | None:
        return col_map.get(name.lower())

    keep_map = {
        "action_number": get_col("actionNumber") or get_col("action_number"),
        "period": get_col("period"),
        "clock": get_col("clock"),
        "action_type": get_col("actionType") or get_col("action_type"),
        "subtype": get_col("subType") or get_col("sub_type"),
        "team_id": get_col("teamId") or get_col("team_id"),
        "person_id": get_col("personId") or get_col("person_id"),
        "points_total": get_col("pointsTotal") or get_col("points_total"),
        "score_home": get_col("scoreHome") or get_col("score_home"),
        "score_away": get_col("scoreAway") or get_col("score_away"),
        "description": get_col("description"),
    }

    selected = [v for v in keep_map.values() if v is not None]
    out = df[selected].copy()

    rename = {v: k for k, v in keep_map.items() if v is not None}
    out = out.rename(columns=rename)

    if "action_number" in out.columns:
        out = out.sort_values("action_number").reset_index(drop=True)

    if "team_id" in out.columns:
        out["team_id"] = out["team_id"].astype(str)
    if "person_id" in out.columns:
        out["person_id"] = out["person_id"].astype(str)

    out["clock_seconds"] = out["clock"].apply(parse_clock_to_seconds) if "clock" in out.columns else 0.0
    return out


def is_possession_ending_event(action_type: str, subtype: str, points_total: int) -> bool:
    action_type = safe_str(action_type).lower()
    subtype = safe_str(subtype).lower()

    if action_type == "turnover":
        return True

    if action_type == "made shot":
        return True

    if action_type == "rebound" and "defensive" in subtype:
        return True

    if action_type == "period" and subtype == "end":
        return True

    if action_type == "free throw" and points_total > 0:
        return True

    return False


def infer_initial_offense_team(pbp_df: pd.DataFrame) -> str:
    for _, row in pbp_df.iterrows():
        team_id = safe_str(row.get("team_id"), "")
        if team_id:
            return team_id
    return ""


def summarize_possession(
    possession_events: list[dict],
    target_player_id: str,
    fallback_offense_team: str = "",
) -> dict:
    if not possession_events:
        return {
            "period": 0,
            "offense_team_id": fallback_offense_team,
            "start_clock": 0.0,
            "end_clock": 0.0,
            "duration": 0.0,
            "num_events": 0,
            "result_type": "empty",
            "points_scored": 0,
            "target_player_involved": 0,
            "target_player_event_count": 0,
            "target_player_scoring_events": 0,
            "score_margin_end": 0,
        }

    first = possession_events[0]
    last = possession_events[-1]

    period = safe_int(first.get("period"), 0)

    offense_team_id = ""
    for ev in possession_events:
        t = safe_str(ev.get("team_id"), "")
        if t:
            offense_team_id = t
            break
    if not offense_team_id:
        offense_team_id = fallback_offense_team

    start_clock = float(first.get("clock_seconds", 0.0))
    end_clock = float(last.get("clock_seconds", 0.0))
    duration = max(0.0, start_clock - end_clock)

    points_scored = 0
    result_type = "other"

    for ev in possession_events:
        action_type = safe_str(ev.get("action_type"), "").lower()
        subtype = safe_str(ev.get("subtype"), "").lower()
        ev_points = safe_int(ev.get("points_total"), 0)

        if action_type == "turnover":
            result_type = "turnover"
        elif action_type == "made shot":
            result_type = "made_shot"
            points_scored = max(points_scored, ev_points)
        elif action_type == "missed shot":
            result_type = "missed_shot"
        elif action_type == "free throw" and ev_points > 0:
            result_type = "free_throw_score"
            points_scored = max(points_scored, ev_points)
        elif action_type == "rebound" and "defensive" in subtype:
            if result_type == "other":
                result_type = "def_rebound_end"

    target_player_event_count = 0
    target_player_scoring_events = 0

    for ev in possession_events:
        person_id = safe_str(ev.get("person_id"), "")
        if person_id == target_player_id:
            target_player_event_count += 1
            action_type = safe_str(ev.get("action_type"), "").lower()
            if action_type in {"made shot", "free throw"}:
                target_player_scoring_events += 1

    target_player_involved = 1 if target_player_event_count > 0 else 0

    score_home = safe_int(last.get("score_home"), 0)
    score_away = safe_int(last.get("score_away"), 0)
    score_margin_end = max(-30, min(30, score_home - score_away))

    return {
        "period": period,
        "offense_team_id": offense_team_id,
        "start_clock": round(start_clock, 2),
        "end_clock": round(end_clock, 2),
        "duration": round(duration, 2),
        "num_events": len(possession_events),
        "result_type": result_type,
        "points_scored": points_scored,
        "target_player_involved": target_player_involved,
        "target_player_event_count": target_player_event_count,
        "target_player_scoring_events": target_player_scoring_events,
        "score_margin_end": score_margin_end,
    }


def build_possession_sequence_for_player_game(
    game_id: str,
    target_player_id: str,
    max_possessions: int = MAX_POSSESSIONS,
) -> list[dict]:
    pbp_df = load_play_by_play(game_id)

    if pbp_df.empty:
        return []

    fallback_offense_team = infer_initial_offense_team(pbp_df)

    possessions = []
    current_events = []

    for _, row in pbp_df.iterrows():
        ev = {
            "period": safe_int(row.get("period"), 0),
            "clock_seconds": float(row.get("clock_seconds", 0.0)),
            "action_type": safe_str(row.get("action_type"), ""),
            "subtype": safe_str(row.get("subtype"), ""),
            "team_id": safe_str(row.get("team_id"), ""),
            "person_id": safe_str(row.get("person_id"), ""),
            "points_total": safe_int(row.get("points_total"), 0),
            "score_home": safe_int(row.get("score_home"), 0),
            "score_away": safe_int(row.get("score_away"), 0),
            "description": safe_str(row.get("description"), ""),
        }

        current_events.append(ev)

        if is_possession_ending_event(ev["action_type"], ev["subtype"], ev["points_total"]):
            possession = summarize_possession(
                current_events,
                target_player_id=target_player_id,
                fallback_offense_team=fallback_offense_team,
            )
            possessions.append(possession)
            current_events = []

    if current_events:
        possession = summarize_possession(
            current_events,
            target_player_id=target_player_id,
            fallback_offense_team=fallback_offense_team,
        )
        possessions.append(possession)

    if max_possessions is not None:
        possessions = possessions[:max_possessions]

    return possessions


def build_possession_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = add_previous_game_id(df)
    out = out.dropna(subset=["previous_game_id"]).reset_index(drop=True)
    out["previous_game_id"] = out["previous_game_id"].astype(str)
    out["player_id"] = out["player_id"].astype(str)

    unique_pairs = out[["previous_game_id", "player_id"]].drop_duplicates().values.tolist()
    print(f"Unique (previous_game_id, player_id) pairs to build: {len(unique_pairs)}")

    seq_cache: dict[tuple[str, str], str] = {}
    failed_pairs: list[str] = []

    for i, pair in enumerate(unique_pairs, start=1):
        previous_game_id, player_id = pair
        key = (str(previous_game_id), str(player_id))
        try:
            seq = build_possession_sequence_for_player_game(
                game_id=str(previous_game_id),
                target_player_id=str(player_id),
            )
            seq_cache[key] = json.dumps(seq, ensure_ascii=False)
            if i % 500 == 0 or i == len(unique_pairs):
                print(f"[{i}/{len(unique_pairs)}] cached possession sequence for game={previous_game_id}, player={player_id}")
        except Exception as exc:
            failed_pairs.append(f"{previous_game_id},{player_id}")
            seq_cache[key] = json.dumps([], ensure_ascii=False)
            print(f"[{i}/{len(unique_pairs)}] FAILED game={previous_game_id}, player={player_id}: {exc}")

    out["possession_sequence_json"] = out.apply(
        lambda row: seq_cache.get((str(row["previous_game_id"]), str(row["player_id"])), "[]"),
        axis=1,
    )

    if failed_pairs:
        failed_path = OUT_PATH.parent / "failed_possession_sequence_pairs.txt"
        failed_path.write_text("\n".join(failed_pairs), encoding="utf-8")
        print(f"Failed possession pairs saved to: {failed_path}")

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
        "possession_sequence_json",
    ]
    final_cols = [c for c in final_cols if c in out.columns]
    return out[final_cols].copy()


def validate_possession_dataset(df: pd.DataFrame) -> None:
    print("Possession dataset shape:", df.shape)
    print("Unique players:", df["player_id"].nunique())
    print("Unique games:", df["game_id"].nunique())

    empty_sequences = (df["possession_sequence_json"] == "[]").sum()
    print("Empty possession sequences:", int(empty_sequences))

    print("\nPreview:")
    print(df.head())

    if len(df) > 0:
        first_seq = json.loads(df.iloc[0]["possession_sequence_json"])
        print("\nFirst possession sequence length:", len(first_seq))
        if first_seq:
            print("First possession token example:")
            print(first_seq[0])


def main() -> None:
    print("Loading feature dataset...")
    df = load_features(FEATURES_PATH)

    print("Building possession-sequence dataset...")
    possession_df = build_possession_dataset(df)

    print("Validating...")
    validate_possession_dataset(possession_df)

    possession_df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved possession sequence dataset to: {OUT_PATH}")


if __name__ == "__main__":
    main()
    