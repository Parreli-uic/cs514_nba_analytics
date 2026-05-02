from __future__ import annotations

from pathlib import Path
import json
import math
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


INPUT_PATH = Path("data/processed/possession_sequences_next_game_points_2025-26_regular_season.parquet")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
FF_DIM = 128
DROPOUT = 0.1
MAX_SEQ_LEN = 512

METRICS_PATH = ARTIFACT_DIR / f"transformer_possessions_metrics_e{EPOCHS}_d{D_MODEL}_s{MAX_SEQ_LEN}.json"
PREDICTIONS_PATH = ARTIFACT_DIR / f"transformer_possessions_predictions_e{EPOCHS}_d{D_MODEL}_s{MAX_SEQ_LEN}.parquet"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.lower() == "nan":
            return default
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def normalize_team_id(value) -> str:
    try:
        if value is None:
            return ""
        if isinstance(value, str) and value.lower() == "nan":
            return ""
        if pd.isna(value):
            return ""
        return str(int(float(value)))
    except Exception:
        return ""


def bucket_duration(duration: float) -> int:
    d = safe_float(duration, 0.0)
    d = max(0.0, min(40.0, d))
    return int(round(d))


def bucket_score_margin(score_margin: int) -> int:
    s = safe_int(score_margin, 0)
    s = max(-30, min(30, s))
    return s + 30  # [0, 60]


def bucket_points_scored(points: int) -> int:
    p = safe_int(points, 0)
    p = max(0, min(4, p))
    return p


def bucket_num_events(num_events: int) -> int:
    n = safe_int(num_events, 0)
    n = max(0, min(20, n))
    return n


def bucket_player_event_count(count: int) -> int:
    c = safe_int(count, 0)
    c = max(0, min(10, c))
    return c


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Possession sequence dataset not found: {path}")

    df = pd.read_parquet(path).copy()
    df.columns = [c.lower() for c in df.columns]

    if "game_date" not in df.columns:
        raise ValueError("game_date is required")

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["player_id"] = df["player_id"].astype(str)
    df["game_id"] = df["game_id"].astype(str)
    return df


def time_based_split(df: pd.DataFrame, split_quantile: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("game_date").reset_index(drop=True)
    split_idx = int(len(df) * split_quantile)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def build_vocabs(train_df: pd.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
    result_vocab = {"<PAD>": 0, "<UNK>": 1}
    offense_team_vocab = {"<PAD>": 0, "<UNK>": 1}

    for seq_json in train_df["possession_sequence_json"]:
        seq = json.loads(seq_json)
        for token in seq:
            result_type = str(token.get("result_type") or "<UNK>").strip()
            offense_team = normalize_team_id(token.get("offense_team_id"))

            if result_type not in result_vocab:
                result_vocab[result_type] = len(result_vocab)

            if offense_team:
                if offense_team not in offense_team_vocab:
                    offense_team_vocab[offense_team] = len(offense_team_vocab)

    return result_vocab, offense_team_vocab


def get_context_feature_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "pts_roll3",
        "pts_roll5",
        "reb_roll3",
        "reb_roll5",
        "ast_roll3",
        "ast_roll5",
        "min_roll3",
        "min_roll5",
        "team_ctx_pace_proxy",
    ]
    return [c for c in candidates if c in df.columns]


def compute_context_stats(train_df: pd.DataFrame, cols: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    means = {}
    stds = {}
    for c in cols:
        series = pd.to_numeric(train_df[c], errors="coerce")
        means[c] = float(series.mean()) if not series.isna().all() else 0.0
        std = float(series.std()) if not series.isna().all() else 1.0
        stds[c] = std if std > 1e-8 else 1.0
    return means, stds


@dataclass
class ParsedRow:
    target: float
    context: list[float]
    result_ids: list[int]
    offense_team_ids: list[int]
    period_ids: list[int]
    duration_ids: list[int]
    score_margin_ids: list[int]
    points_scored_ids: list[int]
    num_events_ids: list[int]
    player_involved_vals: list[float]
    player_event_count_ids: list[int]
    player_scoring_event_ids: list[int]


def parse_row(
    row: pd.Series,
    result_vocab: dict[str, int],
    offense_team_vocab: dict[str, int],
    context_cols: list[str],
    context_means: dict[str, float],
    context_stds: dict[str, float],
) -> ParsedRow:
    seq = json.loads(row["possession_sequence_json"])

    result_ids = []
    offense_team_ids = []
    period_ids = []
    duration_ids = []
    score_margin_ids = []
    points_scored_ids = []
    num_events_ids = []
    player_involved_vals = []
    player_event_count_ids = []
    player_scoring_event_ids = []

    for token in seq[:MAX_SEQ_LEN]:
        result_type = str(token.get("result_type") or "<UNK>").strip()
        offense_team = normalize_team_id(token.get("offense_team_id"))

        result_ids.append(result_vocab.get(result_type, result_vocab["<UNK>"]))
        offense_team_ids.append(offense_team_vocab.get(offense_team, offense_team_vocab["<UNK>"]))

        period = safe_int(token.get("period"), 0)
        period = max(0, min(10, period))
        period_ids.append(period)

        duration_ids.append(bucket_duration(token.get("duration")))
        score_margin_ids.append(bucket_score_margin(token.get("score_margin_end")))
        points_scored_ids.append(bucket_points_scored(token.get("points_scored")))
        num_events_ids.append(bucket_num_events(token.get("num_events")))
        player_involved_vals.append(float(safe_int(token.get("target_player_involved"), 0)))
        player_event_count_ids.append(bucket_player_event_count(token.get("target_player_event_count")))
        player_scoring_event_ids.append(bucket_points_scored(token.get("target_player_scoring_events")))

    context = []
    for c in context_cols:
        value = safe_float(row.get(c), context_means[c])
        value = (value - context_means[c]) / context_stds[c]
        context.append(value)

    target = float(row["target_next_game_pts"])

    return ParsedRow(
        target=target,
        context=context,
        result_ids=result_ids,
        offense_team_ids=offense_team_ids,
        period_ids=period_ids,
        duration_ids=duration_ids,
        score_margin_ids=score_margin_ids,
        points_scored_ids=points_scored_ids,
        num_events_ids=num_events_ids,
        player_involved_vals=player_involved_vals,
        player_event_count_ids=player_event_count_ids,
        player_scoring_event_ids=player_scoring_event_ids,
    )


class PossessionSequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        result_vocab: dict[str, int],
        offense_team_vocab: dict[str, int],
        context_cols: list[str],
        context_means: dict[str, float],
        context_stds: dict[str, float],
    ) -> None:
        self.rows = [
            parse_row(
                row=row,
                result_vocab=result_vocab,
                offense_team_vocab=offense_team_vocab,
                context_cols=context_cols,
                context_means=context_means,
                context_stds=context_stds,
            )
            for _, row in df.iterrows()
        ]
        self.context_dim = len(context_cols)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> ParsedRow:
        return self.rows[idx]


def collate_batch(batch: list[ParsedRow]) -> dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_len = min(MAX_SEQ_LEN, max(len(x.result_ids) for x in batch))

    result_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    offense_team_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    period_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    duration_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    score_margin_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    points_scored_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    num_events_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    player_involved_vals = torch.zeros((batch_size, max_len), dtype=torch.float32)
    player_event_count_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    player_scoring_event_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    context = torch.tensor([x.context for x in batch], dtype=torch.float32)
    targets = torch.tensor([x.target for x in batch], dtype=torch.float32)

    for i, row in enumerate(batch):
        seq_len = min(len(row.result_ids), max_len)

        result_ids[i, :seq_len] = torch.tensor(row.result_ids[:seq_len], dtype=torch.long)
        offense_team_ids[i, :seq_len] = torch.tensor(row.offense_team_ids[:seq_len], dtype=torch.long)
        period_ids[i, :seq_len] = torch.tensor(row.period_ids[:seq_len], dtype=torch.long)
        duration_ids[i, :seq_len] = torch.tensor(row.duration_ids[:seq_len], dtype=torch.long)
        score_margin_ids[i, :seq_len] = torch.tensor(row.score_margin_ids[:seq_len], dtype=torch.long)
        points_scored_ids[i, :seq_len] = torch.tensor(row.points_scored_ids[:seq_len], dtype=torch.long)
        num_events_ids[i, :seq_len] = torch.tensor(row.num_events_ids[:seq_len], dtype=torch.long)
        player_involved_vals[i, :seq_len] = torch.tensor(row.player_involved_vals[:seq_len], dtype=torch.float32)
        player_event_count_ids[i, :seq_len] = torch.tensor(row.player_event_count_ids[:seq_len], dtype=torch.long)
        player_scoring_event_ids[i, :seq_len] = torch.tensor(row.player_scoring_event_ids[:seq_len], dtype=torch.long)
        attention_mask[i, :seq_len] = True

    return {
        "result_ids": result_ids,
        "offense_team_ids": offense_team_ids,
        "period_ids": period_ids,
        "duration_ids": duration_ids,
        "score_margin_ids": score_margin_ids,
        "points_scored_ids": points_scored_ids,
        "num_events_ids": num_events_ids,
        "player_involved_vals": player_involved_vals,
        "player_event_count_ids": player_event_count_ids,
        "player_scoring_event_ids": player_scoring_event_ids,
        "attention_mask": attention_mask,
        "context": context,
        "targets": targets,
    }


class PossessionTransformerRegressor(nn.Module):
    def __init__(
        self,
        result_vocab_size: int,
        offense_team_vocab_size: int,
        context_dim: int,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_layers: int = NUM_LAYERS,
        ff_dim: int = FF_DIM,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        self.result_emb = nn.Embedding(result_vocab_size, d_model, padding_idx=0)
        self.offense_team_emb = nn.Embedding(offense_team_vocab_size, d_model, padding_idx=0)
        self.period_emb = nn.Embedding(11, d_model, padding_idx=0)
        self.duration_emb = nn.Embedding(41, d_model, padding_idx=0)
        self.score_margin_emb = nn.Embedding(61, d_model)
        self.points_scored_emb = nn.Embedding(5, d_model)
        self.num_events_emb = nn.Embedding(21, d_model)
        self.player_event_count_emb = nn.Embedding(11, d_model)
        self.player_scoring_event_emb = nn.Embedding(5, d_model)

        self.player_involved_proj = nn.Linear(1, d_model)
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.regressor = nn.Sequential(
            nn.Linear(d_model + 32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        result_ids: torch.Tensor,
        offense_team_ids: torch.Tensor,
        period_ids: torch.Tensor,
        duration_ids: torch.Tensor,
        score_margin_ids: torch.Tensor,
        points_scored_ids: torch.Tensor,
        num_events_ids: torch.Tensor,
        player_involved_vals: torch.Tensor,
        player_event_count_ids: torch.Tensor,
        player_scoring_event_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = result_ids.shape
        positions = torch.arange(seq_len, device=result_ids.device).unsqueeze(0).expand(batch_size, seq_len)

        x = (
            self.result_emb(result_ids)
            + self.offense_team_emb(offense_team_ids)
            + self.period_emb(period_ids)
            + self.duration_emb(duration_ids)
            + self.score_margin_emb(score_margin_ids)
            + self.points_scored_emb(points_scored_ids)
            + self.num_events_emb(num_events_ids)
            + self.player_event_count_emb(player_event_count_ids)
            + self.player_scoring_event_emb(player_scoring_event_ids)
            + self.player_involved_proj(player_involved_vals.unsqueeze(-1))
            + self.pos_emb(positions)
        )

        src_key_padding_mask = ~attention_mask
        encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        mask = attention_mask.unsqueeze(-1).float()
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        context_repr = self.context_mlp(context)
        fused = torch.cat([pooled, context_repr], dim=1)

        preds = self.regressor(fused).squeeze(-1)
        return preds


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            preds = model(
                result_ids=batch["result_ids"],
                offense_team_ids=batch["offense_team_ids"],
                period_ids=batch["period_ids"],
                duration_ids=batch["duration_ids"],
                score_margin_ids=batch["score_margin_ids"],
                points_scored_ids=batch["points_scored_ids"],
                num_events_ids=batch["num_events_ids"],
                player_involved_vals=batch["player_involved_vals"],
                player_event_count_ids=batch["player_event_count_ids"],
                player_scoring_event_ids=batch["player_scoring_event_ids"],
                attention_mask=batch["attention_mask"],
                context=batch["context"],
            )

            preds_all.append(preds.cpu().numpy())
            targets_all.append(batch["targets"].cpu().numpy())

    preds = np.concatenate(preds_all)
    targets = np.concatenate(targets_all)

    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    return mae, rmse, preds, targets


def main() -> None:
    set_seed()

    print("Loading possession sequence dataset...")
    df = load_dataset(INPUT_PATH)

    print("Creating time-based split...")
    train_df, test_df = time_based_split(df)

    print("Building vocabularies from train set...")
    result_vocab, offense_team_vocab = build_vocabs(train_df)

    context_cols = get_context_feature_columns(df)
    context_means, context_stds = compute_context_stats(train_df, context_cols)

    print("Building datasets...")
    train_ds = PossessionSequenceDataset(
        train_df,
        result_vocab,
        offense_team_vocab,
        context_cols,
        context_means,
        context_stds,
    )
    test_ds = PossessionSequenceDataset(
        test_df,
        result_vocab,
        offense_team_vocab,
        context_cols,
        context_means,
        context_stds,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PossessionTransformerRegressor(
        result_vocab_size=len(result_vocab),
        offense_team_vocab_size=len(offense_team_vocab),
        context_dim=train_ds.context_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    print("Training possession transformer...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            preds = model(
                result_ids=batch["result_ids"],
                offense_team_ids=batch["offense_team_ids"],
                period_ids=batch["period_ids"],
                duration_ids=batch["duration_ids"],
                score_margin_ids=batch["score_margin_ids"],
                points_scored_ids=batch["points_scored_ids"],
                num_events_ids=batch["num_events_ids"],
                player_involved_vals=batch["player_involved_vals"],
                player_event_count_ids=batch["player_event_count_ids"],
                player_scoring_event_ids=batch["player_scoring_event_ids"],
                attention_mask=batch["attention_mask"],
                context=batch["context"],
            )

            loss = loss_fn(preds, batch["targets"])
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))
        test_mae, test_rmse, _, _ = evaluate_model(model, test_loader, device)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test MAE: {test_mae:.4f} | "
            f"Test RMSE: {test_rmse:.4f}"
        )

    final_mae, final_rmse, preds, targets = evaluate_model(model, test_loader, device)

    pred_df = test_df.copy().reset_index(drop=True)
    pred_df["predicted_pts"] = preds
    pred_df["actual_pts"] = targets
    pred_df["error"] = pred_df["predicted_pts"] - pred_df["actual_pts"]
    pred_df["abs_error"] = pred_df["error"].abs()
    pred_df.to_parquet(PREDICTIONS_PATH, index=False)

    metrics = {
        "model_type": "possession_transformer",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "mae": final_mae,
        "rmse": final_rmse,
        "device": str(device),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "num_layers": NUM_LAYERS,
        "ff_dim": FF_DIM,
        "dropout": DROPOUT,
        "max_seq_len": MAX_SEQ_LEN,
        "result_vocab_size": len(result_vocab),
        "offense_team_vocab_size": len(offense_team_vocab),
        "context_dim": train_ds.context_dim,
        "train_date_min": str(train_df["game_date"].min()),
        "train_date_max": str(train_df["game_date"].max()),
        "test_date_min": str(test_df["game_date"].min()),
        "test_date_max": str(test_df["game_date"].max()),
        "predictions_path": str(PREDICTIONS_PATH),
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nPossession transformer metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics to: {METRICS_PATH}")
    print(f"Saved predictions to: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()