"""
Long-Term Load Forecasting - Traditional ML Methods Comparison
Compares 7 algorithms on BuildingsBench Electricity dataset (UCI LD2011-2014).
Train: 2011-2013, Test: 2014 (multi-step horizon: day-ahead 24h)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import time

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT   = "/data/buildings_bench/BuildingsBench/Electricity"
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FORECAST_HORIZON = 24          # predict next 24 hours (day-ahead)
N_BUILDINGS      = 10          # use first N buildings for speed
RANDOM_STATE     = 42

# ── Load & merge data ─────────────────────────────────────────────────────────
print("Loading Electricity dataset (2011–2014)...")
dfs = []
for year in [2011, 2012, 2013, 2014]:
    path = os.path.join(DATA_ROOT, f"LD2011_2014_clean={year}.csv")
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    dfs.append(df)
data = pd.concat(dfs, axis=0).sort_index()

# keep first N buildings, drop any NaN
buildings = data.columns[:N_BUILDINGS].tolist()
data = data[buildings].dropna()
print(f"  Loaded {len(data)} hourly rows, {len(buildings)} buildings: {data.index[0]} → {data.index[-1]}")

# ── Feature engineering ────────────────────────────────────────────────────────
def make_features(df: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """
    For each building, create supervised samples:
      X = [lags 1..48, rolling mean 24h/168h, calendar features]
      y = load at t+horizon
    Returns a single DataFrame with multi-index (building, time).
    """
    rows = []
    for bld in df.columns:
        s = df[bld].copy()
        feat = pd.DataFrame(index=s.index)

        # Calendar
        feat["hour"]       = s.index.hour
        feat["dow"]        = s.index.dayofweek
        feat["month"]      = s.index.month
        feat["is_weekend"] = (s.index.dayofweek >= 5).astype(int)

        # Lag features
        for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
            feat[f"lag_{lag}"] = s.shift(lag)

        # Rolling statistics
        feat["roll_mean_24"]  = s.shift(1).rolling(24).mean()
        feat["roll_std_24"]   = s.shift(1).rolling(24).std()
        feat["roll_mean_168"] = s.shift(1).rolling(168).mean()

        # Target
        feat["target"] = s.shift(-horizon)
        feat["building"] = bld
        rows.append(feat)

    combined = pd.concat(rows)
    combined = combined.dropna()
    return combined

print("Engineering features...")
all_feat = make_features(data, horizon=FORECAST_HORIZON)

# ── Train / test split (temporal) ─────────────────────────────────────────────
feature_cols = [c for c in all_feat.columns if c not in ("target", "building")]
cutoff = pd.Timestamp("2014-01-01")

train = all_feat[all_feat.index < cutoff]
test  = all_feat[all_feat.index >= cutoff]

X_train, y_train = train[feature_cols].values, train["target"].values
X_test,  y_test  = test[feature_cols].values,  test["target"].values
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── Models ────────────────────────────────────────────────────────────────────
MODELS = {
    "Linear Regression":    Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "Ridge Regression":     Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
    "Lasso Regression":     Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.1, max_iter=5000))]),
    "KNN (k=10)":           Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=10, n_jobs=-1))]),
    "Random Forest":        RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE),
    "Extra Trees":          ExtraTreesRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE),
    "Gradient Boosting":    GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                                       subsample=0.8, random_state=RANDOM_STATE),
}

# ── Train & evaluate ──────────────────────────────────────────────────────────
print("\nTraining and evaluating models...")
results = []
predictions = {}

for name, model in MODELS.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)          # energy can't be negative

    mae   = mean_absolute_error(y_test, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
    r2    = r2_score(y_test, y_pred)
    mape  = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100
    cv    = rmse / (y_test.mean() + 1e-6) * 100   # CV-RMSE %

    results.append(dict(Model=name, MAE=mae, RMSE=rmse, MAPE=mape,
                        CV_RMSE=cv, R2=r2, Train_Time_s=round(train_time, 1)))
    predictions[name] = y_pred
    print(f"  {name:<22}  MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.1f}%  R²={r2:.3f}  ({train_time:.0f}s)")

results_df = pd.DataFrame(results).sort_values("RMSE")

# ── Save metrics CSV ──────────────────────────────────────────────────────────
csv_path = os.path.join(OUTPUT_DIR, "ml_forecast_metrics.csv")
results_df.to_csv(csv_path, index=False, float_format="%.4f")
print(f"\nMetrics saved → {csv_path}")

# ── Plot 1: Metrics bar chart ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Long-Term Load Forecasting – Traditional ML Comparison\n"
             "Electricity Dataset (UCI LD2011-2014), 24h Day-Ahead Horizon", fontsize=13)

palette = sns.color_palette("tab10", len(results_df))
models_sorted = results_df["Model"].tolist()

for ax, metric in zip(axes, ["MAE", "RMSE", "MAPE"]):
    vals = results_df[metric].values
    bars = ax.barh(models_sorted, vals, color=palette)
    ax.set_xlabel(metric + (" (%)" if metric == "MAPE" else " (kWh)"), fontsize=11)
    ax.set_title(metric, fontsize=12)
    ax.invert_yaxis()
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "ml_metrics_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Plot 2: R² and CV-RMSE ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Model Performance – R² and CV-RMSE", fontsize=13)

r2_sorted = results_df.sort_values("R2", ascending=False)
ax = axes[0]
bars = ax.barh(r2_sorted["Model"], r2_sorted["R2"], color=palette)
ax.set_xlabel("R²", fontsize=11)
ax.set_title("R² Score (higher = better)", fontsize=12)
ax.invert_yaxis()
ax.axvline(0, color="gray", linewidth=0.8)
for bar, v in zip(bars, r2_sorted["R2"]):
    ax.text(max(bar.get_width(), 0) + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}", va="center", fontsize=8)

cv_sorted = results_df.sort_values("CV_RMSE")
ax = axes[1]
bars = ax.barh(cv_sorted["Model"], cv_sorted["CV_RMSE"], color=palette)
ax.set_xlabel("CV-RMSE (%)", fontsize=11)
ax.set_title("CV-RMSE % (lower = better)", fontsize=12)
ax.invert_yaxis()
for bar, v in zip(bars, cv_sorted["CV_RMSE"]):
    ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
            f"{v:.1f}%", va="center", fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "ml_r2_cvrmse.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Plot 3: Time-series forecast sample (one building, one week) ──────────────
SAMPLE_BUILDING = buildings[0]
mask_test = (test.index >= pd.Timestamp("2014-06-01")) & \
            (test.index <  pd.Timestamp("2014-06-08")) & \
            (test["building"] == SAMPLE_BUILDING)
sample_idx  = test.index[mask_test]
sample_true = y_test[mask_test.values]

fig, axes = plt.subplots(len(MODELS), 1, figsize=(14, 3 * len(MODELS)), sharex=True)
fig.suptitle(f"Day-Ahead Forecast vs Actual – {SAMPLE_BUILDING} (Jun 2014, 1 week)", fontsize=13)

for ax, (name, y_pred_all) in zip(axes, predictions.items()):
    pred_sample = y_pred_all[mask_test.values]
    ax.plot(sample_idx, sample_true, color="black", lw=1.5, label="Actual")
    ax.plot(sample_idx, pred_sample, color="tomato", lw=1.2, linestyle="--", label=name)
    ax.set_ylabel("kWh", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    mae_s = mean_absolute_error(sample_true, pred_sample)
    ax.set_title(f"{name}  (MAE={mae_s:.2f})", fontsize=9)

axes[-1].set_xlabel("Time", fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "ml_forecast_timeseries.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Plot 4: Scatter actual vs predicted ──────────────────────────────────────
# sample 3000 points for readability
np.random.seed(42)
idx_sample = np.random.choice(len(y_test), size=min(3000, len(y_test)), replace=False)

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
fig.suptitle("Actual vs Predicted (random 3000 test samples)", fontsize=13)

for ax, (name, y_pred_all) in zip(axes, predictions.items()):
    yt = y_test[idx_sample]
    yp = y_pred_all[idx_sample]
    ax.scatter(yt, yp, alpha=0.15, s=5, color="steelblue")
    lim = max(yt.max(), yp.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Actual (kWh)", fontsize=8)
    ax.set_ylabel("Predicted (kWh)", fontsize=8)
    r2 = r2_score(yt, yp)
    ax.set_title(f"{name}\nR²={r2:.3f}", fontsize=9)

axes[-1].set_visible(False)  # hide extra subplot
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "ml_scatter_actual_pred.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Plot 5: Feature importance (tree models) ──────────────────────────────────
tree_models = {k: v for k, v in MODELS.items()
               if hasattr(v, "feature_importances_")}

if tree_models:
    fig, axes = plt.subplots(1, len(tree_models), figsize=(6 * len(tree_models), 5))
    if len(tree_models) == 1:
        axes = [axes]
    fig.suptitle("Top-15 Feature Importances (Tree-based Models)", fontsize=13)

    for ax, (name, model) in zip(axes, tree_models.items()):
        imp = pd.Series(model.feature_importances_, index=feature_cols)
        top = imp.nlargest(15).sort_values()
        ax.barh(top.index, top.values, color="steelblue")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Importance", fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "ml_feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()

# ── Plot 6: Summary radar / heatmap ──────────────────────────────────────────
metrics_norm = results_df.set_index("Model")[["MAE", "RMSE", "MAPE", "CV_RMSE", "R2"]].copy()
# normalise to 0-1 (lower-is-better for errors, higher-is-better for R2)
for col in ["MAE", "RMSE", "MAPE", "CV_RMSE"]:
    mn, mx = metrics_norm[col].min(), metrics_norm[col].max()
    metrics_norm[col] = 1 - (metrics_norm[col] - mn) / (mx - mn + 1e-9)   # invert: 1=best
mn, mx = metrics_norm["R2"].min(), metrics_norm["R2"].max()
metrics_norm["R2"] = (metrics_norm["R2"] - mn) / (mx - mn + 1e-9)

fig, ax = plt.subplots(figsize=(9, 5))
sns.heatmap(metrics_norm.T, annot=True, fmt=".2f", cmap="YlGn",
            linewidths=0.5, ax=ax, vmin=0, vmax=1)
ax.set_title("Normalised Performance Heatmap\n(1 = best, 0 = worst per metric)", fontsize=12)
ax.set_xlabel("Model", fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "ml_performance_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("FINAL RESULTS (sorted by RMSE)")
print("=" * 65)
print(results_df[["Model", "MAE", "RMSE", "MAPE", "CV_RMSE", "R2", "Train_Time_s"]].to_string(index=False))
print("\nOutput files:")
for f in ["ml_forecast_metrics.csv", "ml_metrics_comparison.png",
          "ml_r2_cvrmse.png", "ml_forecast_timeseries.png",
          "ml_scatter_actual_pred.png", "ml_feature_importance.png",
          "ml_performance_heatmap.png"]:
    print(f"  results/{f}")
print("Done.")
