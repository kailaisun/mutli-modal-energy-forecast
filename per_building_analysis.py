"""
Per-Building Metrics Analysis
Computes and visualises per-building MAE, CV-RMSE, MAPE, R² for all 7 ML models.
Reveals how mixed-scale data distorts aggregate CV-RMSE.
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

DATA_ROOT  = "/data/buildings_bench/BuildingsBench/Electricity"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FORECAST_HORIZON = 24
N_BUILDINGS      = 10
RANDOM_STATE     = 42

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
dfs = []
for year in [2011, 2012, 2013, 2014]:
    df = pd.read_csv(os.path.join(DATA_ROOT, f"LD2011_2014_clean={year}.csv"),
                     parse_dates=["timestamp"], index_col="timestamp")
    dfs.append(df)
data = pd.concat(dfs).sort_index()
buildings = data.columns[:N_BUILDINGS].tolist()
data = data[buildings].dropna()

# ── Feature engineering ───────────────────────────────────────────────────────
def make_features(df, horizon=24):
    rows = []
    for bld in df.columns:
        s = df[bld].copy()
        feat = pd.DataFrame(index=s.index)
        feat["hour"]       = s.index.hour
        feat["dow"]        = s.index.dayofweek
        feat["month"]      = s.index.month
        feat["is_weekend"] = (s.index.dayofweek >= 5).astype(int)
        for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
            feat[f"lag_{lag}"] = s.shift(lag)
        feat["roll_mean_24"]  = s.shift(1).rolling(24).mean()
        feat["roll_std_24"]   = s.shift(1).rolling(24).std()
        feat["roll_mean_168"] = s.shift(1).rolling(168).mean()
        feat["target"]   = s.shift(-horizon)
        feat["building"] = bld
        rows.append(feat)
    return pd.concat(rows).dropna()

print("Engineering features...")
all_feat = make_features(data, FORECAST_HORIZON)
feature_cols = [c for c in all_feat.columns if c not in ("target", "building")]
cutoff = pd.Timestamp("2014-01-01")
train = all_feat[all_feat.index < cutoff]
test  = all_feat[all_feat.index >= cutoff]
X_train, y_train = train[feature_cols].values, train["target"].values
X_test,  y_test  = test[feature_cols].values,  test["target"].values

# ── Models ────────────────────────────────────────────────────────────────────
MODELS = {
    "Linear Regression": Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]),
    "Ridge Regression":  Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
    "Lasso Regression":  Pipeline([("sc", StandardScaler()), ("m", Lasso(alpha=0.1, max_iter=5000))]),
    "KNN (k=10)":        Pipeline([("sc", StandardScaler()), ("m", KNeighborsRegressor(n_neighbors=10, n_jobs=-1))]),
    "Random Forest":     RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE),
    "Extra Trees":       ExtraTreesRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                                    subsample=0.8, random_state=RANDOM_STATE),
}

print("Training models...")
predictions = {}
for name, model in MODELS.items():
    model.fit(X_train, y_train)
    predictions[name] = np.clip(model.predict(X_test), 0, None)
    print(f"  {name} done")

# ── Per-building metrics ──────────────────────────────────────────────────────
print("\nComputing per-building metrics...")
test_buildings = test["building"].values
records = []
for name, y_pred in predictions.items():
    for bld in buildings:
        mask = test_buildings == bld
        yt = y_test[mask]
        yp = y_pred[mask]
        bld_mean = yt.mean()
        mae   = mean_absolute_error(yt, yp)
        rmse  = np.sqrt(mean_squared_error(yt, yp))
        r2    = r2_score(yt, yp)
        mape  = np.mean(np.abs((yt - yp) / (yt + 1e-6))) * 100
        cvrmse = rmse / (bld_mean + 1e-6) * 100
        records.append(dict(Model=name, Building=bld,
                            Mean_kWh=round(bld_mean, 1),
                            MAE=mae, RMSE=rmse, MAPE=mape,
                            CV_RMSE=cvrmse, R2=r2))

pb = pd.DataFrame(records)
pb.to_csv(os.path.join(OUTPUT_DIR, "ml_per_building_metrics.csv"), index=False, float_format="%.4f")

# building info (mean consumption label)
bld_means = {bld: data[bld].mean() for bld in buildings}
bld_labels = {bld: f"{bld}\n(mean={bld_means[bld]:.0f})" for bld in buildings}

# ── Plot 1: CV-RMSE per building per model (heatmap) ─────────────────────────
pivot_cv = pb.pivot_table(index="Model", columns="Building", values="CV_RMSE")
pivot_cv = pivot_cv[sorted(buildings, key=lambda b: bld_means[b])]   # sort by consumption

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(pivot_cv, annot=True, fmt=".1f", cmap="YlOrRd",
            linewidths=0.4, ax=ax, cbar_kws={"label": "CV-RMSE (%)"})
ax.set_title("Per-Building CV-RMSE (%) — sorted by mean consumption (low → high)\n"
             "Low-consumption buildings inflate aggregate CV-RMSE", fontsize=12)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticklabels([f"{b}\n({bld_means[b]:.0f} kWh)" for b in pivot_cv.columns],
                   fontsize=8, rotation=30, ha="right")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "pb_cvrmse_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Plot 2: R² per building per model (heatmap) ───────────────────────────────
pivot_r2 = pb.pivot_table(index="Model", columns="Building", values="R2")
pivot_r2 = pivot_r2[sorted(buildings, key=lambda b: bld_means[b])]

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(pivot_r2, annot=True, fmt=".3f", cmap="YlGn",
            linewidths=0.4, ax=ax, vmin=0.85, vmax=1.0,
            cbar_kws={"label": "R²"})
ax.set_title("Per-Building R² — sorted by mean consumption (low → high)", fontsize=12)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticklabels([f"{b}\n({bld_means[b]:.0f} kWh)" for b in pivot_r2.columns],
                   fontsize=8, rotation=30, ha="right")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "pb_r2_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Plot 3: CV-RMSE vs mean consumption scatter ───────────────────────────────
palette = sns.color_palette("tab10", len(MODELS))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Why Aggregate CV-RMSE is Misleading: Scale Effect", fontsize=13)

ax = axes[0]
for (name, color) in zip(MODELS.keys(), palette):
    sub = pb[pb["Model"] == name]
    ax.scatter(sub["Mean_kWh"], sub["CV_RMSE"], label=name, color=color, alpha=0.8, s=60)
ax.set_xlabel("Building Mean Consumption (kWh)", fontsize=11)
ax.set_ylabel("CV-RMSE (%)", fontsize=11)
ax.set_title("CV-RMSE vs Building Scale\n(smaller buildings → higher CV-RMSE)", fontsize=11)
ax.legend(fontsize=7, loc="upper right")
ax.set_xscale("log")

ax = axes[1]
for (name, color) in zip(MODELS.keys(), palette):
    sub = pb[pb["Model"] == name]
    ax.scatter(sub["Mean_kWh"], sub["R2"], label=name, color=color, alpha=0.8, s=60)
ax.set_xlabel("Building Mean Consumption (kWh)", fontsize=11)
ax.set_ylabel("R²", fontsize=11)
ax.set_title("R² vs Building Scale\n(consistent across scales)", fontsize=11)
ax.legend(fontsize=7, loc="lower right")
ax.set_xscale("log")

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "pb_scale_effect.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Plot 4: Per-building model ranking (best model per building) ──────────────
best_model = pb.loc[pb.groupby("Building")["CV_RMSE"].idxmin(), ["Building", "Model", "CV_RMSE", "R2"]]
worst_model = pb.loc[pb.groupby("Building")["CV_RMSE"].idxmax(), ["Building", "Model", "CV_RMSE", "R2"]]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Best vs Worst Model Per Building (by CV-RMSE)", fontsize=13)

bld_order = sorted(buildings, key=lambda b: bld_means[b])
model_colors = {m: c for m, c in zip(MODELS.keys(), palette)}

for ax, title, df_rank in [(axes[0], "Best Model (lowest CV-RMSE)", best_model),
                            (axes[1], "Worst Model (highest CV-RMSE)", worst_model)]:
    df_rank = df_rank.set_index("Building").reindex(bld_order)
    colors = [model_colors[m] for m in df_rank["Model"]]
    bars = ax.barh([f"{b}\n({bld_means[b]:.0f} kWh)" for b in bld_order],
                   df_rank["CV_RMSE"].values, color=colors)
    for bar, (_, row) in zip(bars, df_rank.iterrows()):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{row['Model']}  ({row['CV_RMSE']:.1f}%)", va="center", fontsize=8)
    ax.set_xlabel("CV-RMSE (%)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, df_rank["CV_RMSE"].max() * 1.6)

# Legend
handles = [plt.Rectangle((0,0),1,1, color=model_colors[m]) for m in MODELS]
axes[0].legend(handles, MODELS.keys(), fontsize=7, loc="lower right")

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "pb_best_worst_model.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Plot 5: Aggregate vs per-building CV-RMSE comparison bar ─────────────────
agg_cv  = pb.groupby("Model")["CV_RMSE"].mean().reset_index()   # mean of per-building
# recompute aggregate (all buildings mixed)
agg_cv2 = []
for name, y_pred in predictions.items():
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cv   = rmse / y_test.mean() * 100
    agg_cv2.append({"Model": name, "Aggregate_CV_RMSE": cv})
agg_cv2 = pd.DataFrame(agg_cv2)
merged = agg_cv.merge(agg_cv2, on="Model").sort_values("CV_RMSE")
merged.columns = ["Model", "Mean_PerBuilding_CVRMSE", "Aggregate_CVRMSE"]

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(merged))
w = 0.35
b1 = ax.bar(x - w/2, merged["Aggregate_CVRMSE"],   w, label="Aggregate CV-RMSE\n(all buildings mixed)",  color="tomato",    alpha=0.85)
b2 = ax.bar(x + w/2, merged["Mean_PerBuilding_CVRMSE"], w, label="Mean per-building CV-RMSE\n(average of individual buildings)", color="steelblue", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(merged["Model"], rotation=20, ha="right", fontsize=9)
ax.set_ylabel("CV-RMSE (%)", fontsize=11)
ax.set_title("Aggregate vs Mean Per-Building CV-RMSE\n"
             "Aggregate is lower because large buildings dominate the mean denominator", fontsize=11)
ax.legend(fontsize=9)
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "pb_aggregate_vs_perbuilding.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Summary table ─────────────────────────────────────────────────────────────
summary = pb.groupby("Model").agg(
    Mean_CVRMSE=("CV_RMSE", "mean"),
    Std_CVRMSE=("CV_RMSE", "std"),
    Mean_R2=("R2", "mean"),
    Std_R2=("R2", "std"),
    Mean_MAPE=("MAPE", "mean"),
).round(3).sort_values("Mean_CVRMSE")
print("\nPer-building summary (mean ± std across buildings):")
print(summary.to_string())

summary.to_csv(os.path.join(OUTPUT_DIR, "ml_per_building_summary.csv"), float_format="%.4f")
print("\nOutput:")
for f in ["pb_cvrmse_heatmap.png","pb_r2_heatmap.png","pb_scale_effect.png",
          "pb_best_worst_model.png","pb_aggregate_vs_perbuilding.png",
          "ml_per_building_metrics.csv","ml_per_building_summary.csv"]:
    print(f"  results/{f}")
print("Done.")
