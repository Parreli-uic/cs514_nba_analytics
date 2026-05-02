from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    XGBOOST_AVAILABLE = False


INPUT_PATH = Path("data/processed/features_next_game_points_2025-26_regular_season.parquet")
MODEL_DIR = Path("artifacts")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH = MODEL_DIR / "baseline_metrics.json"


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {path}")

    df = pd.read_parquet(path).copy()
    df.columns = [c.lower() for c in df.columns]

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])
    if "target_next_game_date" in df.columns:
        df["target_next_game_date"] = pd.to_datetime(df["target_next_game_date"])

    return df


def build_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str], str]:
    target_col = "target_next_game_pts"

    numeric_features = [
        "is_home",
        "wl_flag",
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
    ]

    categorical_features = [
        "team_abbreviation",
        "opponent_abbreviation",
        "home_away",
    ]

    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    return numeric_features, categorical_features, target_col


def time_based_split(df: pd.DataFrame, split_quantile: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "game_date" not in df.columns:
        raise ValueError("game_date column is required for time-based split")

    df = df.sort_values("game_date").reset_index(drop=True)
    split_idx = int(len(df) * split_quantile)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    return train_df, test_df


def build_model():
    if XGBOOST_AVAILABLE:
        return XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )

    return GradientBoostingRegressor(random_state=42)


def main() -> None:
    print("Loading feature dataset...")
    df = load_features(INPUT_PATH)

    numeric_features, categorical_features, target_col = build_feature_lists(df)

    feature_cols = numeric_features + categorical_features
    df = df.dropna(subset=[target_col]).copy()

    train_df, test_df = time_based_split(df)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = build_model()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    print("Training baseline model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating...")
    preds = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    metrics = {
        "model_type": "xgboost" if XGBOOST_AVAILABLE else "gradient_boosting_fallback",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count_numeric": len(numeric_features),
        "feature_count_categorical": len(categorical_features),
        "mae": float(mae),
        "rmse": float(rmse),
        "train_date_min": str(train_df["game_date"].min()),
        "train_date_max": str(train_df["game_date"].max()),
        "test_date_min": str(test_df["game_date"].min()),
        "test_date_max": str(test_df["game_date"].max()),
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nBaseline metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()