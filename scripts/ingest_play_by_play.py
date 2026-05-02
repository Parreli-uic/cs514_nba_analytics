from __future__ import annotations

from pathlib import Path
import time
import traceback
import pandas as pd

from nba_api.live.nba.endpoints import playbyplay


SEASON = "2025-26"
SEASON_TYPE = "Regular Season"
SLEEP_SECONDS = 0.8
OVERWRITE = False

INTERIM_DIR = Path("data/interim")
RAW_PBP_DIR = Path("data/raw/play_by_play")
RAW_PBP_DIR.mkdir(parents=True, exist_ok=True)


def normalize_label(value: str) -> str:
    return value.lower().replace(" ", "_")


def get_game_ids_path(season: str, season_type: str) -> Path:
    return INTERIM_DIR / f"game_ids_{season}_{normalize_label(season_type)}.parquet"


def get_output_dir(season: str, season_type: str) -> Path:
    outdir = RAW_PBP_DIR / season / normalize_label(season_type)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def load_game_ids(season: str, season_type: str) -> pd.DataFrame:
    path = get_game_ids_path(season, season_type)
    if not path.exists():
        raise FileNotFoundError(f"Game ID file not found: {path}")

    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    df["game_id"] = df["game_id"].astype(str)

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])

    return df


def fetch_play_by_play(game_id: str) -> pd.DataFrame:
    response = playbyplay.PlayByPlay(game_id=game_id)
    data = response.get_dict()

    actions = data.get("game", {}).get("actions", [])
    if not actions:
        raise ValueError(f"No play-by-play actions returned for game_id={game_id}")

    df = pd.DataFrame(actions)
    df["game_id"] = game_id
    return df


def save_raw_pbp(
    df: pd.DataFrame,
    game_id: str,
    outdir: Path,
    overwrite: bool = False,
) -> Path:
    outpath = outdir / f"{game_id}.parquet"

    if outpath.exists() and not overwrite:
        return outpath

    df.to_parquet(outpath, index=False)
    return outpath


def main() -> None:
    print(f"Loading game IDs for {SEASON} | {SEASON_TYPE}...")
    games_df = load_game_ids(SEASON, SEASON_TYPE)
    outdir = get_output_dir(SEASON, SEASON_TYPE)

    total_games = len(games_df)
    success_count = 0
    skipped_count = 0
    failed_games: list[str] = []

    print(f"Found {total_games} games.")
    print(f"Saving raw play-by-play files to: {outdir}")

    for i, row in games_df.iterrows():
        game_id = row["game_id"]
        outpath = outdir / f"{game_id}.parquet"

        if outpath.exists() and not OVERWRITE:
            skipped_count += 1
            print(f"[{i + 1}/{total_games}] skipped existing file for game_id={game_id}")
            continue

        try:
            pbp_df = fetch_play_by_play(game_id)
            save_raw_pbp(pbp_df, game_id, outdir, overwrite=OVERWRITE)
            success_count += 1

            print(
                f"[{i + 1}/{total_games}] saved game_id={game_id} "
                f"rows={len(pbp_df)}"
            )

            time.sleep(SLEEP_SECONDS)

        except Exception as exc:
            failed_games.append(game_id)
            print(f"[{i + 1}/{total_games}] FAILED game_id={game_id}: {exc}")
            traceback.print_exc()
            time.sleep(SLEEP_SECONDS)

    print("\nIngestion complete.")
    print(f"Successful pulls: {success_count}")
    print(f"Skipped existing: {skipped_count}")
    print(f"Failed pulls: {len(failed_games)}")

    if failed_games:
        failed_path = outdir / "failed_game_ids.txt"
        failed_path.write_text("\n".join(failed_games), encoding="utf-8")
        print(f"Failed game IDs written to: {failed_path}")


if __name__ == "__main__":
    main()