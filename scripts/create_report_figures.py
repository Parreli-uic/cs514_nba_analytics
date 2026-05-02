import pandas as pd
import matplotlib.pyplot as plt
import json
import os

os.makedirs("artifacts/figures", exist_ok=True)

# =========================
# LOAD DATA
# =========================
pred_path = "artifacts/transformer_possessions_predictions_e10_d128_s512.parquet"
df = pd.read_parquet(pred_path)

# =========================
# 1. PREDICTED VS ACTUAL
# =========================
plt.figure()
plt.scatter(df["actual_pts"], df["predicted_pts"], alpha=0.3)
plt.xlabel("Actual Points")
plt.ylabel("Predicted Points")
plt.title("Predicted vs Actual Points")

# diagonal line
min_val = min(df["actual_pts"].min(), df["predicted_pts"].min())
max_val = max(df["actual_pts"].max(), df["predicted_pts"].max())
plt.plot([min_val, max_val], [min_val, max_val])

plt.savefig("artifacts/figures/predicted_vs_actual.png")
plt.close()

# =========================
# 2. ERROR DISTRIBUTION
# =========================
plt.figure()
plt.hist(df["abs_error"], bins=50)
plt.xlabel("Absolute Error")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Error")

plt.savefig("artifacts/figures/error_distribution.png")
plt.close()

# =========================
# 3. MODEL COMPARISON
# =========================

with open("artifacts/baseline_metrics.json") as f:
    baseline = json.load(f)

with open("artifacts/transformer_metrics.json") as f:
    transformer = json.load(f)

with open("artifacts/transformer_possessions_metrics_e10_d128_s512.json") as f:
    possession = json.load(f)

models = ["XGBoost", "Transformer (Events)", "Transformer (Possessions)"]
mae = [baseline["mae"], transformer["mae"], possession["mae"]]
rmse = [baseline["rmse"], transformer["rmse"], possession["rmse"]]

# MAE chart
plt.figure()
plt.bar(models, mae)
plt.ylabel("MAE")
plt.title("Model Comparison (MAE)")
plt.xticks(rotation=15)

plt.savefig("artifacts/figures/mae_comparison.png")
plt.close()

# RMSE chart
plt.figure()
plt.bar(models, rmse)
plt.ylabel("RMSE")
plt.title("Model Comparison (RMSE)")
plt.xticks(rotation=15)

plt.savefig("artifacts/figures/rmse_comparison.png")
plt.close()

print("Figures saved to artifacts/figures/")