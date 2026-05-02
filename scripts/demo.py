import pandas as pd

df = pd.read_parquet(
    "artifacts/transformer_possessions_predictions_e10_d128_s512.parquet"
)

demo = df[[
    "player_name",
    "target_next_game_id",
    "predicted_pts",
    "actual_pts",
    "abs_error"
]].rename(columns={
    "target_next_game_id": "game_id"
})

print("Sample predictions:")
print(demo.head(10))

print("\nBest predictions:")
print(demo.sort_values("abs_error").head(10))

print("\nWorst predictions:")
print(demo.sort_values("abs_error", ascending=False).head(10))