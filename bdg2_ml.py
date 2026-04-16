"""
BDG-2 ML Baseline – 7 traditional models
20 buildings sampled from Bear/Fox/Rat (seed=42)
Train: 2016  |  Test: 2017  |  Horizon: 24h  |  Lookback features: lags + rolling + calendar
"""

import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
BDG2_ROOT  = "/data/buildings_bench/BuildingsBench/BDG-2"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results", "bdg2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FORECAST_HORIZON = 24
RANDOM_STATE     = 42

# 20 buildings sampled uniformly from Bear(7) / Fox(7) / Rat(6), seed=42
SELECTED = {
    "Bear": [
        "Bear_education_Lila", "Bear_public_Orville", "Bear_education_Zandra",
        "Bear_education_Herb", "Bear_education_Iris", "Bear_public_Rayna",
        "Bear_lodging_Dannie",
    ],
    "Fox": [
        "Fox_assembly_Cathy", "Fox_assembly_Audrey", "Fox_office_Susanne",
        "Fox_education_Yolande", "Fox_education_Tonya", "Fox_education_Henrietta",
        "Fox_assembly_Lakeisha",
    ],
    "Rat": [
        "Rat_public_Sharron", "Rat_education_Pat", "Rat_office_Lora",
        "Rat_assembly_Cristina", "Rat_education_Adell", "Rat_education_Nellie",
    ],
}

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading BDG-2 data (2016–2017)...")
site_dfs = []
for site, blds in SELECTED.items():
    frames = []
    for year in [2016, 2017]:
        path = os.path.join(BDG2_ROOT, f"{site}_clean={year}.csv")
        df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp",
                         usecols=["timestamp"] + blds)
        frames.append(df)
    site_dfs.append(pd.concat(frames).sort_index())
data = pd.concat(site_dfs, axis=1).dropna()
buildings = [b for blds in SELECTED.values() for b in blds]
cutoff = pd.Timestamp("2017-01-01")
print(f"  {len(data)} rows, {len(buildings)} buildings: {data.index[0]} → {data.index[-1]}")

# ── Feature engineering ────────────────────────────────────────────────────────
def make_features(df, horizon=24):
    rows = []
    for bld in df.columns:
        s = df[bld].copy()
        feat = pd.DataFrame(index=s.index)
        feat["hour"]        = s.index.hour
        feat["dow"]         = s.index.dayofweek
        feat["month"]       = s.index.month
        feat["is_weekend"]  = (s.index.dayofweek >= 5).astype(int)
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
train = all_feat[all_feat.index < cutoff]
test  = all_feat[all_feat.index >= cutoff]
X_train, y_train = train[feature_cols].values, train["target"].values
X_test,  y_test  = test[feature_cols].values,  test["target"].values
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── Models ─────────────────────────────────────────────────────────────────────
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

# ── Train & evaluate ───────────────────────────────────────────────────────────
print("\nTraining ML models...")
results, predictions = [], {}
for name, model in MODELS.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    t1 = time.time()
    y_pred = np.clip(model.predict(X_test), 0, None)
    mae   = mean_absolute_error(y_test, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
    r2    = r2_score(y_test, y_pred)
    mape  = np.mean(np.abs((y_test - y_pred) / (np.abs(y_test) + 1e-6))) * 100
    cv    = rmse / (y_test.mean() + 1e-6) * 100
    results.append(dict(Model=name, MAE=mae, RMSE=rmse, MAPE=mape,
                        CV_RMSE=cv, R2=r2, Train_Time_s=round(t1-t0, 1)))
    predictions[name] = y_pred
    print(f"  {name:<22}  MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.1f}%  R²={r2:.3f}  ({t1-t0:.0f}s)")

results_df = pd.DataFrame(results).sort_values("RMSE")
results_df.to_csv(os.path.join(OUTPUT_DIR, "bdg2_ml_metrics.csv"), index=False, float_format="%.4f")

# ── Plots ──────────────────────────────────────────────────────────────────────
palette = sns.color_palette("tab10", len(results_df))
models_sorted = results_df["Model"].tolist()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("BDG-2 ML Baseline – 24h Day-Ahead Load Forecasting\n"
             "20 Buildings (Bear/Fox/Rat), Train:2016 Test:2017", fontsize=12)
for ax, metric in zip(axes, ["MAE", "RMSE", "MAPE"]):
    vals = results_df[metric].values
    bars = ax.barh(models_sorted, vals, color=palette)
    ax.set_xlabel(metric + (" (%)" if metric == "MAPE" else " (kWh)"), fontsize=11)
    ax.set_title(metric, fontsize=12); ax.invert_yaxis()
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2,
                f"{v:.2f}", va="center", fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "bdg2_ml_metrics_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()

# Feature importance
tree_models = {k: v for k, v in MODELS.items() if hasattr(v, "feature_importances_")}
if tree_models:
    fig, axes = plt.subplots(1, len(tree_models), figsize=(6*len(tree_models), 5))
    if len(tree_models) == 1: axes = [axes]
    fig.suptitle("Top-15 Feature Importances (BDG-2)", fontsize=13)
    for ax, (name, model) in zip(axes, tree_models.items()):
        imp = pd.Series(model.feature_importances_, index=feature_cols)
        top = imp.nlargest(15).sort_values()
        ax.barh(top.index, top.values, color="steelblue")
        ax.set_title(name, fontsize=11); ax.set_xlabel("Importance", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "bdg2_ml_feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()

print("\n" + "="*65)
print("BDG-2 ML RESULTS (sorted by RMSE)")
print("="*65)
print(results_df[["Model","MAE","RMSE","MAPE","CV_RMSE","R2","Train_Time_s"]].to_string(index=False))
print(f"\nSaved → {OUTPUT_DIR}/bdg2_ml_metrics.csv")
print("Done.")
