"""
Long-Term Load Forecasting – Deep Learning Models Comparison
Models: LSTM, Transformer, TCN, PatchTST, MambaFormer
Fix: per-building normalisation on both X and y; inverse-transform at evaluation.
"""

import os, time, math, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT   = "/data/buildings_bench/BuildingsBench/Electricity"
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN     = 168        # 1 week lookback
HORIZON     = 24         # predict t+24h
N_BUILDINGS = 10
BATCH_SIZE  = 512
EPOCHS      = 40
LR          = 1e-3
PATIENCE    = 6
print(f"Device: {DEVICE}")

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading data...")
dfs = []
for year in [2011, 2012, 2013, 2014]:
    df = pd.read_csv(os.path.join(DATA_ROOT, f"LD2011_2014_clean={year}.csv"),
                     parse_dates=["timestamp"], index_col="timestamp")
    dfs.append(df)
data = pd.concat(dfs).sort_index()
buildings = data.columns[:N_BUILDINGS].tolist()
data = data[buildings].dropna()
cutoff = pd.Timestamp("2014-01-01")

# Per-building z-score (fit on train only)
scalers = {}
train_sc = {}; test_sc = {}
for bld in buildings:
    sc = StandardScaler()
    tr = data.loc[data.index < cutoff, bld].values.reshape(-1, 1)
    te = data.loc[data.index >= cutoff, bld].values.reshape(-1, 1)
    train_sc[bld] = sc.fit_transform(tr).ravel()
    test_sc[bld]  = sc.transform(te).ravel()
    scalers[bld]  = sc

# ── Dataset ───────────────────────────────────────────────────────────────────
class LoadDataset(Dataset):
    """Normalised X and y; also stores bld_idx for inverse transform."""
    def __init__(self, scaled_dict, seq_len, horizon):
        self.samples = []   # (x_norm, y_norm, bld_idx)
        self.bld_list = list(scaled_dict.keys())
        for bidx, bld in enumerate(self.bld_list):
            v = scaled_dict[bld].astype(np.float32)
            for i in range(seq_len, len(v) - horizon):
                x = v[i - seq_len: i]
                y = v[i + horizon - 1]
                self.samples.append((x, y, bidx))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, y, b = self.samples[idx]
        return (torch.from_numpy(x).unsqueeze(-1),
                torch.tensor(y, dtype=torch.float32),
                b)

train_ds = LoadDataset(train_sc, SEQ_LEN, HORIZON)
test_ds  = LoadDataset(test_sc,  SEQ_LEN, HORIZON)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
print(f"  Train: {len(train_ds):,}  |  Test: {len(test_ds):,}")

def inverse_transform(preds_norm, bld_indices, buildings, scalers):
    """Convert normalised predictions back to kWh."""
    out = np.empty_like(preds_norm)
    for bidx, bld in enumerate(buildings):
        mask = bld_indices == bidx
        if mask.any():
            out[mask] = scalers[bld].inverse_transform(
                preds_norm[mask].reshape(-1, 1)).ravel()
    return out

# ── Models ────────────────────────────────────────────────────────────────────

# 1. LSTM
class LSTMModel(nn.Module):
    def __init__(self, hidden=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


# 2. Vanilla Transformer
class TransformerModel(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=3, dim_ff=256, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.pos  = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.enc  = nn.TransformerEncoder(enc, num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        x = self.proj(x) + self.pos
        return self.head(self.enc(x)[:, -1]).squeeze(-1)


# 3. TCN
class CausalConv1d(nn.Module):
    def __init__(self, ch, k=3, d=1):
        super().__init__()
        self.pad  = (k - 1) * d
        self.conv = nn.Conv1d(ch, ch, k, dilation=d, groups=ch)  # depthwise
        self.pw   = nn.Conv1d(ch, ch, 1)

    def forward(self, x):
        return self.pw(F.gelu(self.conv(F.pad(x, (self.pad, 0)))))

class TCNModel(nn.Module):
    def __init__(self, channels=64, nlayers=6, k=3):
        super().__init__()
        self.proj   = nn.Conv1d(1, channels, 1)
        self.blocks = nn.ModuleList([
            nn.Sequential(CausalConv1d(channels, k, 2**i),
                          nn.BatchNorm1d(channels),
                          nn.GELU())
            for i in range(nlayers)])
        self.head = nn.Sequential(nn.Linear(channels, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):                         # (B,T,1)
        x = self.proj(x.permute(0,2,1))           # (B,C,T)
        for blk in self.blocks:
            x = x + blk(x)
        return self.head(x[:, :, -1]).squeeze(-1)


# 4. PatchTST
class PatchTST(nn.Module):
    def __init__(self, patch_len=24, stride=12, d_model=64, nhead=4, nlayers=3, dim_ff=256, dropout=0.1):
        super().__init__()
        n_patches = (SEQ_LEN - patch_len) // stride + 1
        self.patch_len = patch_len
        self.stride    = stride
        self.proj = nn.Linear(patch_len, d_model)
        self.pos  = nn.Parameter(torch.zeros(1, n_patches, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.enc  = nn.TransformerEncoder(enc, nlayers)
        self.head = nn.Sequential(nn.Linear(d_model * n_patches, 128),
                                  nn.ReLU(), nn.Linear(128, 1))
        self.n_patches = n_patches

    def forward(self, x):                         # (B,T,1)
        x = x.squeeze(-1)                         # (B,T)
        patches = x.unfold(1, self.patch_len, self.stride)   # (B,np,pl)
        patches = self.proj(patches) + self.pos
        out = self.enc(patches).flatten(1)        # (B, np*d)
        return self.head(out).squeeze(-1)


# 5. MambaFormer – vectorised parallel scan Mamba + Transformer
class MambaBlock(nn.Module):
    """
    Efficient Mamba block using a parallel (associative) scan.
    The recurrence h_t = a_t * h_{t-1} + b_t is computed via log-space
    prefix scan, which avoids Python loops.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        D = d_model * expand
        self.D_inner  = D
        self.d_state  = d_state
        self.in_proj  = nn.Linear(d_model, D * 2)
        self.conv1d   = nn.Conv1d(D, D, d_conv, padding=d_conv-1, groups=D)
        self.x_proj   = nn.Linear(D, d_state * 2 + D)  # B, C, delta
        self.dt_proj  = nn.Linear(D, D, bias=True)
        self.A_log    = nn.Parameter(
            torch.log(torch.arange(1, d_state+1, dtype=torch.float)
                      .unsqueeze(0).expand(D, -1)))    # (D, d_state)
        self.D_skip   = nn.Parameter(torch.ones(D))
        self.out_proj = nn.Linear(D, d_model)
        self.norm     = nn.LayerNorm(d_model)
        self.drop     = nn.Dropout(dropout)

    @staticmethod
    def _parallel_scan(a, b):
        """
        Compute h_t = a_t * h_{t-1} + b_t for all t in parallel.
        a, b: (B, T, D, S)
        Uses log-space trick: log_a cumsum then exp multiplication.
        """
        # Convert to log space for numerical stability
        log_a  = torch.log(torch.clamp(a, min=1e-6))
        log_a_cumsum = torch.cumsum(log_a, dim=1)          # (B,T,D,S)
        # b_scaled_i = b_i * exp(sum_{j=i+1}^{T} log_a_j) applied via scan
        # Use sequential approx via: h_t = exp(cumsum_a[:t]) * sum_i b_i/exp(cumsum_a[:i])
        scales   = torch.exp(log_a_cumsum)                  # (B,T,D,S)
        b_scaled = b / (scales + 1e-8)
        h = scales * torch.cumsum(b_scaled, dim=1)          # (B,T,D,S)
        return h

    def forward(self, x):                                   # (B,T,d_model)
        B, T, _ = x.shape
        residual = x
        xz  = self.in_proj(x)                               # (B,T,2D)
        x_, z = xz.chunk(2, dim=-1)                         # (B,T,D)

        # Depthwise conv
        x_ = x_.permute(0,2,1)                              # (B,D,T)
        x_ = self.conv1d(x_)[:,:,:T].permute(0,2,1)        # (B,T,D)
        x_ = F.silu(x_)

        # SSM params
        bcd  = self.x_proj(x_)                              # (B,T, 2S+D)
        B_s  = bcd[:,:,:self.d_state]                       # (B,T,S)
        C_s  = bcd[:,:,self.d_state:2*self.d_state]         # (B,T,S)
        dt   = F.softplus(self.dt_proj(bcd[:,:,2*self.d_state:]))  # (B,T,D)

        A = -torch.exp(self.A_log)                           # (D,S)
        # Discretise: dA = exp(dt * A), dB = dt * B
        dA = torch.exp(dt.unsqueeze(-1) * A)                # (B,T,D,S)
        dB = (dt.unsqueeze(-1) *
              B_s.unsqueeze(2).expand(-1,-1,self.D_inner,-1))  # (B,T,D,S)
        u  = x_.unsqueeze(-1).expand(-1,-1,-1,self.d_state)    # (B,T,D,S)

        # Parallel scan: h_t = dA_t * h_{t-1} + dB_t * u_t
        h = self._parallel_scan(dA, dB * u)                 # (B,T,D,S)

        # Output
        y = (h * C_s.unsqueeze(2)).sum(-1)                  # (B,T,D)
        y = y + self.D_skip * x_
        y = y * F.silu(z)
        y = self.drop(self.out_proj(y))
        return self.norm(y + residual)


class MambaFormer(nn.Module):
    """Alternating Mamba + TransformerEncoder blocks."""
    def __init__(self, d_model=64, d_state=16, nhead=4,
                 n_layers=3, dim_ff=256, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.pos  = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(MambaBlock(d_model, d_state=d_state, dropout=dropout))
            self.layers.append(
                nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        x = self.proj(x) + self.pos
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x)[:, -1]).squeeze(-1)


# ── Training ──────────────────────────────────────────────────────────────────
def run(model, name):
    model = model.to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit  = nn.HuberLoss(delta=1.0)

    best_loss, best_state, patience_cnt = float("inf"), None, 0
    t_losses, v_losses = [], []
    t0 = time.time()

    for ep in range(1, EPOCHS+1):
        model.train()
        run_loss = 0.
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += loss.item() * len(xb)
        t_losses.append(run_loss / len(train_ds))

        model.eval()
        val_loss = 0.
        with torch.no_grad():
            for xb, yb, _ in test_loader:
                val_loss += crit(model(xb.to(DEVICE)), yb.to(DEVICE)).item() * len(xb)
        val_loss /= len(test_ds)
        v_losses.append(val_loss)
        sched.step()

        if ep % 5 == 0:
            print(f"    ep {ep:3d}/{EPOCHS}  train={t_losses[-1]:.4f}  val={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss, patience_cnt = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"    Early stop ep {ep}")
                break

    model.load_state_dict(best_state)
    elapsed = time.time() - t0

    # Evaluate + inverse transform
    model.eval()
    preds_n, trues_n, bld_ids = [], [], []
    with torch.no_grad():
        for xb, yb, bidx in test_loader:
            preds_n.append(model(xb.to(DEVICE)).cpu().numpy())
            trues_n.append(yb.numpy())
            bld_ids.append(bidx.numpy())
    preds_n = np.concatenate(preds_n)
    trues_n = np.concatenate(trues_n)
    bld_ids = np.concatenate(bld_ids)

    preds_kWh = inverse_transform(preds_n, bld_ids, buildings, scalers)
    trues_kWh = inverse_transform(trues_n, bld_ids, buildings, scalers)
    preds_kWh = np.clip(preds_kWh, 0, None)

    mae   = mean_absolute_error(trues_kWh, preds_kWh)
    rmse  = np.sqrt(mean_squared_error(trues_kWh, preds_kWh))
    r2    = r2_score(trues_kWh, preds_kWh)
    mape  = np.mean(np.abs((trues_kWh - preds_kWh) / (trues_kWh + 1e-6))) * 100
    cvrmse = rmse / (trues_kWh.mean() + 1e-6) * 100
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  → MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.1f}%  R²={r2:.4f}  ({elapsed:.0f}s)")
    metrics = dict(Model=name, MAE=mae, RMSE=rmse, MAPE=mape, CV_RMSE=cvrmse,
                   R2=r2, Params_M=round(n_params/1e6, 2), Train_Time_s=round(elapsed,1))
    return metrics, preds_kWh, trues_kWh, bld_ids, (t_losses, v_losses)


# ── Run all models ─────────────────────────────────────────────────────────────
MODEL_DEFS = {
    "LSTM":        LSTMModel(hidden=128, num_layers=2),
    "Transformer": TransformerModel(d_model=64, nhead=4, num_layers=3),
    "TCN":         TCNModel(channels=64, nlayers=6),
    "PatchTST":    PatchTST(patch_len=24, stride=12, d_model=64, nhead=4, nlayers=3),
    "MambaFormer": MambaFormer(d_model=64, d_state=16, nhead=4, n_layers=3),
}

all_results, all_preds, all_loss_curves = [], {}, {}
for name, model in MODEL_DEFS.items():
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*55}\n  {name}  ({n/1e6:.2f}M params)\n{'='*55}")
    m, preds, trues, bids, curves = run(model, name)
    all_results.append(m)
    all_preds[name] = (preds, trues, bids)
    all_loss_curves[name] = curves

results_df = pd.DataFrame(all_results).sort_values("RMSE")
results_df.to_csv(os.path.join(OUTPUT_DIR, "dl_forecast_metrics.csv"), index=False, float_format="%.4f")

print("\n" + "="*60)
print("DEEP LEARNING RESULTS (sorted by RMSE)")
print("="*60)
print(results_df[["Model","MAE","RMSE","MAPE","CV_RMSE","R2","Params_M","Train_Time_s"]].to_string(index=False))

# ── Figures ───────────────────────────────────────────────────────────────────
palette_dl = sns.color_palette("Set2", len(results_df))

# Fig 1: Loss curves
fig, axes = plt.subplots(1, len(MODEL_DEFS), figsize=(5*len(MODEL_DEFS), 4))
fig.suptitle("Training / Validation Loss Curves (Huber)", fontsize=13)
for ax, (name, (tl, vl)) in zip(axes, all_loss_curves.items()):
    ax.plot(tl, label="Train"); ax.plot(vl, label="Val")
    ax.set_title(name, fontsize=10); ax.set_xlabel("Epoch"); ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "dl_loss_curves.png"), dpi=150, bbox_inches="tight")
plt.close()

# Fig 2: DL metrics bar
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Deep Learning Models – 24h Day-Ahead Load Forecasting", fontsize=13)
for ax, metric in zip(axes, ["MAE","RMSE","MAPE"]):
    bars = ax.barh(results_df["Model"], results_df[metric], color=palette_dl)
    ax.set_xlabel(metric + (" (%)" if metric=="MAPE" else " (kWh)"), fontsize=11)
    ax.set_title(metric, fontsize=12); ax.invert_yaxis()
    for bar, v in zip(bars, results_df[metric]):
        ax.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2,
                f"{v:.2f}", va="center", fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "dl_metrics_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()

# Fig 3: ML vs DL combined
ml_path = os.path.join(OUTPUT_DIR, "ml_forecast_metrics.csv")
if os.path.exists(ml_path):
    ml_df = pd.read_csv(ml_path)
    ml_df["Type"] = "Traditional ML"
    dl_df = results_df[["Model","MAE","RMSE","MAPE","CV_RMSE","R2"]].copy()
    dl_df["Type"] = "Deep Learning"
    combined = pd.concat([ml_df[["Model","MAE","RMSE","MAPE","CV_RMSE","R2","Type"]],
                          dl_df], ignore_index=True).sort_values("RMSE")
    combined.to_csv(os.path.join(OUTPUT_DIR, "ml_vs_dl_metrics.csv"), index=False, float_format="%.4f")

    type_c = {"Traditional ML":"steelblue","Deep Learning":"tomato"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle("Traditional ML vs Deep Learning\nUCI Electricity Dataset, 24h Day-Ahead Horizon", fontsize=13)
    for ax, metric in zip(axes, ["MAE","RMSE","MAPE"]):
        colors = [type_c[t] for t in combined["Type"]]
        bars   = ax.barh(combined["Model"], combined[metric], color=colors)
        ax.set_xlabel(metric + (" (%)" if metric=="MAPE" else " (kWh)"), fontsize=11)
        ax.set_title(metric, fontsize=12); ax.invert_yaxis()
        for bar, v in zip(bars, combined[metric]):
            ax.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2,
                    f"{v:.2f}", va="center", fontsize=8)
    from matplotlib.patches import Patch
    axes[-1].legend(handles=[Patch(facecolor="steelblue",label="Traditional ML"),
                              Patch(facecolor="tomato",label="Deep Learning")],
                    loc="lower right", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "ml_vs_dl_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 4: Heatmap all models
    norm = combined.set_index("Model")[["MAE","RMSE","MAPE","CV_RMSE","R2"]].copy()
    for c in ["MAE","RMSE","MAPE","CV_RMSE"]:
        mn,mx = norm[c].min(),norm[c].max()
        norm[c] = 1-(norm[c]-mn)/(mx-mn+1e-9)
    mn,mx = norm["R2"].min(),norm["R2"].max()
    norm["R2"] = (norm["R2"]-mn)/(mx-mn+1e-9)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(norm.T, annot=True, fmt=".2f", cmap="YlGn",
                linewidths=0.5, ax=ax, vmin=0, vmax=1)
    n_ml = ml_df.shape[0]
    ax.axvline(n_ml, color="navy", lw=2.5, linestyle="--", label="ML | DL boundary")
    ax.set_title("Normalised Performance Heatmap – All Models (1=best)\n"
                 "Dashed line: Traditional ML | Deep Learning", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "all_models_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()

# Fig 5: Time series sample (first 168 test points)
n_show = 168
_, trues_ref, _ = list(all_preds.values())[0]
fig, axes = plt.subplots(len(MODEL_DEFS), 1, figsize=(14, 3.5*len(MODEL_DEFS)), sharex=True)
fig.suptitle("Day-Ahead Forecast vs Actual – First Week of Test Set", fontsize=13)
for ax, (name, (preds, trues, _)) in zip(axes, all_preds.items()):
    ax.plot(trues[:n_show], color="black", lw=1.5, label="Actual")
    ax.plot(preds[:n_show], color="tomato", lw=1.2, ls="--", label=name)
    ax.set_title(f"{name}  (MAE={mean_absolute_error(trues[:n_show],preds[:n_show]):.2f} kWh)", fontsize=9)
    ax.set_ylabel("kWh", fontsize=8); ax.legend(fontsize=8)
axes[-1].set_xlabel("Hours", fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "dl_forecast_timeseries.png"), dpi=150, bbox_inches="tight")
plt.close()

# Fig 6: Scatter
idx_s = np.random.choice(len(trues_ref), size=min(3000, len(trues_ref)), replace=False)
fig, axes = plt.subplots(1, len(MODEL_DEFS), figsize=(5*len(MODEL_DEFS), 5))
fig.suptitle("Actual vs Predicted (3000 random test samples)", fontsize=13)
for ax, (name, (preds, trues, _)) in zip(axes, all_preds.items()):
    yt,yp = trues[idx_s], preds[idx_s]
    ax.scatter(yt,yp,alpha=0.15,s=5,color="steelblue")
    lim = max(yt.max(),yp.max())*1.05
    ax.plot([0,lim],[0,lim],"r--",lw=1)
    ax.set_xlim(0,lim); ax.set_ylim(0,lim)
    ax.set_xlabel("Actual (kWh)",fontsize=8); ax.set_ylabel("Predicted (kWh)",fontsize=8)
    ax.set_title(f"{name}\nR²={r2_score(yt,yp):.3f}",fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "dl_scatter_actual_pred.png"), dpi=150, bbox_inches="tight")
plt.close()

print("\nAll figures saved. Done.")
