from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

GAME_LOGS_RAW_DIR = RAW_DIR / "game_logs"
PLAY_BY_PLAY_RAW_DIR = RAW_DIR / "play_by_play"
TEAM_CONTEXT_RAW_DIR = RAW_DIR / "team_context"
