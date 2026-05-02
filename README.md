# CS 514 NBA Player Performance Project

Sequence-based modeling of NBA player performance using transformer architectures, with comparison against gradient boosting baselines.

## Project layout

```text
cs514_nba_project/
├── data/
│   ├── raw/
│   │   ├── game_logs/
│   │   ├── play_by_play/
│   │   └── team_context/
│   ├── interim/
│   └── processed/
├── docs/
├── notebooks/
├── scripts/
├── src/
│   └── cs514_nba_project/
│       ├── ingest/
│       ├── preprocess/
│       ├── features/
│       ├── models/
│       └── utils/
└── tests/
```

## First milestone

Build a reproducible ingestion and preprocessing pipeline for NBA regular-season:
- player game logs
- play-by-play events
- team context data

## Suggested run order

1. `python scripts/ingest_player_logs.py`
2. `python scripts/ingest_play_by_play.py`
3. `python scripts/ingest_team_context.py`
4. `python scripts/clean_player_logs.py`
5. `python scripts/clean_play_by_play.py`
6. `python scripts/clean_team_context.py`

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pip install -e .
```
