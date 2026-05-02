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

SEED = 42
BATCH_SIZE = 64
# EPOCHS = 5
EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
# D_MODEL = 64
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
FF_DIM = 128
DROPOUT = 0.1
# MAX_SEQ_LEN = 256
MAX_SEQ_LEN = 512

INPUT_PATH = Path("data/processed/event_sequences_next_game_points_2025-26_regular_season.parquet")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH = ARTIFACT_DIR / f"transformer_metrics_e{EPOCHS}_d{D_MODEL}_s{MAX_SEQ_LEN}.json"

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_clock_to_seconds(clock_str: str | None) -> float:
    if clock_str is None:
        return 0.0
    if not isinstance(clock_str, str):
        return 0.0

    # Example format: PT12M00.00S
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


def bucket_score_diff(score_home, score_away) -> int:
    diff = safe_int(score_home, 0) - safe_int(score_away, 0)
    diff = max(-30, min(30, diff))
    return diff + 30  # maps [-30,30] -> [0,60]


def bucket_points_total(points_total) -> int:
    pts = safe_int(points_total, 0)
    pts = max(0, min(4, pts))
    return pts  # 0..4


@dataclass
class ParsedRow:
    game_date: pd.Timestamp
    target: float
    context: list[float]
    action_ids: list[int]
    subtype_ids: list[int]
    period_ids: list[int]
    score_diff_ids: list[int]
    points_bucket_ids: list[int]
    clock_vals: list[float]


def load_event_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Event sequence dataset not found: {path}")

    df = pd.read_parquet(path).copy()
    df.columns = [c.lower() for c in df.columns]

    if "game_date" not in df.columns:
        raise ValueError("game_date is required in the event dataset")

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
    action_vocab = {"<PAD>": 0, "<UNK>": 1}
    subtype_vocab = {"<PAD>": 0, "<UNK>": 1}

    for seq_json in train_df["event_sequence_json"]:
        events = json.loads(seq_json)
        for event in events:
            action = str(event.get("action_type") or "<UNK>").strip()
            subtype = str(event.get("subtype") or "<UNK>").strip()

            if action not in action_vocab:
                action_vocab[action] = len(action_vocab)
            if subtype not in subtype_vocab:
                subtype_vocab[subtype] = len(subtype_vocab)

    return action_vocab, subtype_vocab


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


def parse_row(
    row: pd.Series,
    action_vocab: dict[str, int],
    subtype_vocab: dict[str, int],
    context_cols: list[str],
    context_means: dict[str, float],
    context_stds: dict[str, float],
) -> ParsedRow:
    events = json.loads(row["event_sequence_json"])

    action_ids = []
    subtype_ids = []
    period_ids = []
    score_diff_ids = []
    points_bucket_ids = []
    clock_vals = []

    for event in events[:MAX_SEQ_LEN]:
        action = str(event.get("action_type") or "<UNK>").strip()
        subtype = str(event.get("subtype") or "<UNK>").strip()

        action_ids.append(action_vocab.get(action, action_vocab["<UNK>"]))
        subtype_ids.append(subtype_vocab.get(subtype, subtype_vocab["<UNK>"]))

        period = safe_int(event.get("period"), 0)
        period = max(0, min(10, period))
        period_ids.append(period)

        score_diff_ids.append(bucket_score_diff(event.get("score_home"), event.get("score_away")))
        points_bucket_ids.append(bucket_points_total(event.get("points_total")))
        clock_vals.append(parse_clock_to_seconds(event.get("clock")) / 720.0)  # normalize by 12 minutes

    context = []
    for c in context_cols:
        value = safe_float(row.get(c), context_means[c])
        value = (value - context_means[c]) / context_stds[c]
        context.append(value)

    target = float(row["target_next_game_pts"])

    return ParsedRow(
        game_date=row["game_date"],
        target=target,
        context=context,
        action_ids=action_ids,
        subtype_ids=subtype_ids,
        period_ids=period_ids,
        score_diff_ids=score_diff_ids,
        points_bucket_ids=points_bucket_ids,
        clock_vals=clock_vals,
    )


class EventSequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        action_vocab: dict[str, int],
        subtype_vocab: dict[str, int],
        context_cols: list[str],
        context_means: dict[str, float],
        context_stds: dict[str, float],
    ) -> None:
        self.rows = [
            parse_row(
                row=row,
                action_vocab=action_vocab,
                subtype_vocab=subtype_vocab,
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
    max_len = min(MAX_SEQ_LEN, max(len(x.action_ids) for x in batch))

    action_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    subtype_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    period_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    score_diff_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    points_bucket_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    clock_vals = torch.zeros((batch_size, max_len), dtype=torch.float32)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    context = torch.tensor([x.context for x in batch], dtype=torch.float32)
    targets = torch.tensor([x.target for x in batch], dtype=torch.float32)

    for i, row in enumerate(batch):
        seq_len = min(len(row.action_ids), max_len)

        action_ids[i, :seq_len] = torch.tensor(row.action_ids[:seq_len], dtype=torch.long)
        subtype_ids[i, :seq_len] = torch.tensor(row.subtype_ids[:seq_len], dtype=torch.long)
        period_ids[i, :seq_len] = torch.tensor(row.period_ids[:seq_len], dtype=torch.long)
        score_diff_ids[i, :seq_len] = torch.tensor(row.score_diff_ids[:seq_len], dtype=torch.long)
        points_bucket_ids[i, :seq_len] = torch.tensor(row.points_bucket_ids[:seq_len], dtype=torch.long)
        clock_vals[i, :seq_len] = torch.tensor(row.clock_vals[:seq_len], dtype=torch.float32)
        attention_mask[i, :seq_len] = True

    return {
        "action_ids": action_ids,
        "subtype_ids": subtype_ids,
        "period_ids": period_ids,
        "score_diff_ids": score_diff_ids,
        "points_bucket_ids": points_bucket_ids,
        "clock_vals": clock_vals,
        "attention_mask": attention_mask,
        "context": context,
        "targets": targets,
    }


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        action_vocab_size: int,
        subtype_vocab_size: int,
        context_dim: int,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_layers: int = NUM_LAYERS,
        ff_dim: int = FF_DIM,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        self.action_emb = nn.Embedding(action_vocab_size, d_model, padding_idx=0)
        self.subtype_emb = nn.Embedding(subtype_vocab_size, d_model, padding_idx=0)
        self.period_emb = nn.Embedding(11, d_model, padding_idx=0)          # 0..10
        self.score_diff_emb = nn.Embedding(61, d_model)                     # 0..60
        self.points_bucket_emb = nn.Embedding(5, d_model)                   # 0..4

        self.clock_proj = nn.Linear(1, d_model)
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
        action_ids: torch.Tensor,
        subtype_ids: torch.Tensor,
        period_ids: torch.Tensor,
        score_diff_ids: torch.Tensor,
        points_bucket_ids: torch.Tensor,
        clock_vals: torch.Tensor,
        attention_mask: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = action_ids.shape
        positions = torch.arange(seq_len, device=action_ids.device).unsqueeze(0).expand(batch_size, seq_len)

        x = (
            self.action_emb(action_ids)
            + self.subtype_emb(subtype_ids)
            + self.period_emb(period_ids)
            + self.score_diff_emb(score_diff_ids)
            + self.points_bucket_emb(points_bucket_ids)
            + self.clock_proj(clock_vals.unsqueeze(-1))
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


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            preds = model(
                action_ids=batch["action_ids"],
                subtype_ids=batch["subtype_ids"],
                period_ids=batch["period_ids"],
                score_diff_ids=batch["score_diff_ids"],
                points_bucket_ids=batch["points_bucket_ids"],
                clock_vals=batch["clock_vals"],
                attention_mask=batch["attention_mask"],
                context=batch["context"],
            )

            preds_all.append(preds.cpu().numpy())
            targets_all.append(batch["targets"].cpu().numpy())

    preds = np.concatenate(preds_all)
    targets = np.concatenate(targets_all)

    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    return mae, rmse


def main() -> None:
    set_seed()

    print("Loading event sequence dataset...")
    df = load_event_dataset(INPUT_PATH)

    print("Creating time-based split...")
    train_df, test_df = time_based_split(df)

    print("Building vocabularies from train set...")
    action_vocab, subtype_vocab = build_vocabs(train_df)

    context_cols = get_context_feature_columns(df)
    context_means, context_stds = compute_context_stats(train_df, context_cols)

    print("Building datasets...")
    train_ds = EventSequenceDataset(
        train_df, action_vocab, subtype_vocab, context_cols, context_means, context_stds
    )
    test_ds = EventSequenceDataset(
        test_df, action_vocab, subtype_vocab, context_cols, context_means, context_stds
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

    model = TransformerRegressor(
        action_vocab_size=len(action_vocab),
        subtype_vocab_size=len(subtype_vocab),
        context_dim=train_ds.context_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    print("Training transformer...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            preds = model(
                action_ids=batch["action_ids"],
                subtype_ids=batch["subtype_ids"],
                period_ids=batch["period_ids"],
                score_diff_ids=batch["score_diff_ids"],
                points_bucket_ids=batch["points_bucket_ids"],
                clock_vals=batch["clock_vals"],
                attention_mask=batch["attention_mask"],
                context=batch["context"],
            )

            loss = loss_fn(preds, batch["targets"])
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))
        test_mae, test_rmse = evaluate_model(model, test_loader, device)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test MAE: {test_mae:.4f} | "
            f"Test RMSE: {test_rmse:.4f}"
        )

    final_mae, final_rmse = evaluate_model(model, test_loader, device)

    metrics = {
        "model_type": "small_transformer",
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
        "action_vocab_size": len(action_vocab),
        "subtype_vocab_size": len(subtype_vocab),
        "context_dim": train_ds.context_dim,
        "train_date_min": str(train_df["game_date"].min()),
        "train_date_max": str(train_df["game_date"].max()),
        "test_date_min": str(test_df["game_date"].min()),
        "test_date_max": str(test_df["game_date"].max()),
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nTransformer metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()