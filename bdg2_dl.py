"""
BDG-2 DL Baseline – PatchTST / Transformer / MambaFormer / TCN
20 buildings sampled from Bear/Fox/Rat (seed=42)
Train: 2016  |  Test: 2017  |  SEQ_LEN=168h  |  HORIZON=24h
Configs identical to UCI experiments.
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
torch.manual_seed(42); np.random.seed(42)
torch.backends.cudnn.enabled = False

# ── Config ─────────────────────────────────────────────────────────────────────
BDG2_ROOT  = "/data/buildings_bench/BuildingsBench/BDG-2"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results", "bdg2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN    = 168
HORIZON    = 24
BATCH_SIZE = 512
EPOCHS     = 40
LR         = 1e-3
PATIENCE   = 6
print(f"Device: {DEVICE}")

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

# ── Data ───────────────────────────────────────────────────────────────────────
print("Loading BDG-2 data (2016–2017)...")
buildings = [b for blds in SELECTED.values() for b in blds]
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
cutoff = pd.Timestamp("2017-01-01")
print(f"  {len(data)} rows, {len(buildings)} buildings")

scalers, train_sc, test_sc = {}, {}, {}
for bld in buildings:
    sc = StandardScaler()
    tr = data.loc[data.index < cutoff, bld].values.reshape(-1, 1)
    te = data.loc[data.index >= cutoff, bld].values.reshape(-1, 1)
    train_sc[bld] = sc.fit_transform(tr).ravel()
    test_sc[bld]  = sc.transform(te).ravel()
    scalers[bld]  = sc

class LoadDataset(Dataset):
    def __init__(self, scaled_dict, seq_len, horizon):
        self.samples = []
        self.bld_list = list(scaled_dict.keys())
        for bidx, bld in enumerate(self.bld_list):
            v = scaled_dict[bld].astype(np.float32)
            for i in range(seq_len, len(v) - horizon):
                self.samples.append((v[i-seq_len:i], v[i+horizon-1], bidx))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, y, b = self.samples[idx]
        return torch.from_numpy(x).unsqueeze(-1), torch.tensor(y, dtype=torch.float32), b

train_ds = LoadDataset(train_sc, SEQ_LEN, HORIZON)
test_ds  = LoadDataset(test_sc,  SEQ_LEN, HORIZON)
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
print(f"  Train: {len(train_ds):,}  |  Test: {len(test_ds):,}")

def inverse_transform(preds_norm, bld_indices):
    out = np.empty_like(preds_norm)
    for bidx, bld in enumerate(buildings):
        mask = bld_indices == bidx
        if mask.any():
            out[mask] = scalers[bld].inverse_transform(preds_norm[mask].reshape(-1,1)).ravel()
    return out

# ── Models ─────────────────────────────────────────────────────────────────────

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


class PatchTST(nn.Module):
    def __init__(self, patch_len=24, stride=12, d_model=64, nhead=4, nlayers=3, dim_ff=256, dropout=0.1):
        super().__init__()
        n_patches = (SEQ_LEN - patch_len) // stride + 1
        self.patch_len = patch_len; self.stride = stride; self.n_patches = n_patches
        self.proj = nn.Linear(patch_len, d_model)
        self.pos  = nn.Parameter(torch.zeros(1, n_patches, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.enc  = nn.TransformerEncoder(enc, nlayers)
        self.head = nn.Sequential(nn.Linear(d_model * n_patches, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x):
        x = x.squeeze(-1)
        patches = x.unfold(1, self.patch_len, self.stride)
        patches = self.proj(patches) + self.pos
        out = self.enc(patches).flatten(1)
        return self.head(out).squeeze(-1)


class TCNBlock(nn.Module):
    def __init__(self, ch, k, d):
        super().__init__()
        self.pad  = (k-1)*d
        self.conv = nn.Conv1d(ch, ch, k, dilation=d, groups=ch)
        self.pw   = nn.Conv1d(ch, ch, 1)
        self.norm = nn.LayerNorm(ch)
    def forward(self, x):
        h = self.pw(F.gelu(self.conv(F.pad(x, (self.pad, 0)))))
        return self.norm((x+h).permute(0,2,1)).permute(0,2,1)

class TCNModel(nn.Module):
    def __init__(self, ch=128, nlayers=6, k=3):
        super().__init__()
        self.proj   = nn.Conv1d(1, ch, 1)
        self.blocks = nn.ModuleList([TCNBlock(ch, k, 2**i) for i in range(nlayers)])
        self.head   = nn.Sequential(nn.Linear(ch, 64), nn.ReLU(), nn.Linear(64, 1))
        self.d = ch
    def forward(self, x):
        h = self.proj(x.permute(0,2,1))
        for b in self.blocks: h = b(h)
        return self.head(h.permute(0,2,1)[:, -1]).squeeze(-1)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        D = d_model * expand
        self.D_inner = D; self.d_state = d_state
        self.in_proj  = nn.Linear(d_model, D*2)
        self.conv1d   = nn.Conv1d(D, D, d_conv, padding=d_conv-1, groups=D)
        self.x_proj   = nn.Linear(D, d_state*2 + D)
        self.dt_proj  = nn.Linear(D, D, bias=True)
        self.A_log    = nn.Parameter(torch.log(
            torch.arange(1, d_state+1, dtype=torch.float).unsqueeze(0).expand(D, -1)))
        self.D_skip   = nn.Parameter(torch.ones(D))
        self.out_proj = nn.Linear(D, d_model)
        self.norm     = nn.LayerNorm(d_model)
        self.drop     = nn.Dropout(dropout)

    @staticmethod
    def _parallel_scan(a, b):
        log_a = torch.log(torch.clamp(a, min=1e-6))
        scales = torch.exp(torch.cumsum(log_a, dim=1))
        h = scales * torch.cumsum(b / (scales + 1e-8), dim=1)
        return h

    def forward(self, x):
        B, T, _ = x.shape
        residual = x
        xz = self.in_proj(x)
        x_, z = xz.chunk(2, dim=-1)
        x_ = self.conv1d(x_.permute(0,2,1))[:,:,:T].permute(0,2,1)
        x_ = F.silu(x_)
        bcd = self.x_proj(x_)
        B_s = bcd[:,:,:self.d_state]
        C_s = bcd[:,:,self.d_state:2*self.d_state]
        dt  = F.softplus(self.dt_proj(bcd[:,:,2*self.d_state:]))
        A   = -torch.exp(self.A_log)
        dA  = torch.exp(dt.unsqueeze(-1) * A)
        dB  = dt.unsqueeze(-1) * B_s.unsqueeze(2).expand(-1,-1,self.D_inner,-1)
        u   = x_.unsqueeze(-1).expand(-1,-1,-1,self.d_state)
        h   = self._parallel_scan(dA, dB * u)
        y   = (h * C_s.unsqueeze(2)).sum(-1) + self.D_skip * x_
        y   = self.drop(self.out_proj(y * F.silu(z)))
        return self.norm(y + residual)

class MambaFormer(nn.Module):
    def __init__(self, d_model=64, d_state=16, nhead=4, n_layers=2, dim_ff=256, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.pos  = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(MambaBlock(d_model, d_state=d_state, dropout=dropout))
            self.layers.append(nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        x = self.proj(x) + self.pos
        for layer in self.layers: x = layer(x)
        return self.head(self.norm(x)[:, -1]).squeeze(-1)

# ── Train / eval ───────────────────────────────────────────────────────────────
def run(model, name):
    model = model.to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit  = nn.HuberLoss(delta=1.0)
    best_loss, best_state, pat = float("inf"), None, 0
    t0 = time.time()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*55}\n  {name}  ({n_params/1e6:.2f}M params)\n{'='*55}")

    for ep in range(1, EPOCHS+1):
        model.train(); tl = 0
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE); opt.zero_grad()
            loss = crit(model(xb), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += loss.item() * len(xb)
        tl /= len(train_ds)
        model.eval(); vl = 0
        with torch.no_grad():
            for xb, yb, _ in test_loader:
                vl += crit(model(xb.to(DEVICE)), yb.to(DEVICE)).item() * len(xb)
        vl /= len(test_ds); sched.step()
        if ep % 5 == 0:
            print(f"    ep {ep:3d}/{EPOCHS}  train={tl:.4f}  val={vl:.4f}", flush=True)
        if vl < best_loss:
            best_loss, pat = vl, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= PATIENCE: print(f"    Early stop ep {ep}"); break

    model.load_state_dict(best_state); model.eval()
    ps, ts, bs = [], [], []
    with torch.no_grad():
        for xb, yb, bidx in test_loader:
            ps.append(model(xb.to(DEVICE)).cpu().numpy())
            ts.append(yb.numpy()); bs.append(bidx.numpy())
    ps = np.concatenate(ps); ts = np.concatenate(ts); bs = np.concatenate(bs)
    pi = np.clip(inverse_transform(ps, bs), 0, None)
    ti = inverse_transform(ts, bs)
    mae    = mean_absolute_error(ti, pi)
    rmse   = np.sqrt(mean_squared_error(ti, pi))
    r2     = r2_score(ti, pi)
    mape   = np.mean(np.abs((ti-pi)/(np.abs(ti)+1e-6)))*100
    cvrmse = rmse / (ti.mean()+1e-6) * 100
    elapsed = time.time() - t0
    print(f"  → MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.1f}%  R²={r2:.4f}  ({elapsed:.0f}s)")
    return dict(Model=name, MAE=round(mae,4), RMSE=round(rmse,4), MAPE=round(mape,4),
                CV_RMSE=round(cvrmse,4), R2=round(r2,6),
                Params_M=round(n_params/1e6,2), Train_Time_s=round(elapsed,1)), pi, ti, bs

# ── Run all ────────────────────────────────────────────────────────────────────
MODEL_DEFS = {
    "PatchTST":    PatchTST(patch_len=24, stride=12, d_model=64, nhead=4, nlayers=3),
    "Transformer": TransformerModel(d_model=64, nhead=4, num_layers=3),
    "MambaFormer": MambaFormer(d_model=64, n_layers=2),
    "TCN":         TCNModel(ch=128, nlayers=6, k=3),
}

all_results, all_preds = [], {}
for name, model in MODEL_DEFS.items():
    m, pi, ti, bs = run(model, name)
    all_results.append(m); all_preds[name] = (pi, ti)

results_df = pd.DataFrame(all_results).sort_values("RMSE")
results_df.to_csv(os.path.join(OUTPUT_DIR, "bdg2_dl_metrics.csv"), index=False, float_format="%.4f")

# ── Plots ──────────────────────────────────────────────────────────────────────
palette = sns.color_palette("Set2", len(results_df))
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("BDG-2 Deep Learning Baseline – 24h Day-Ahead Load Forecasting\n"
             "20 Buildings (Bear/Fox/Rat), Train:2016 Test:2017", fontsize=12)
for ax, metric in zip(axes, ["MAE", "RMSE", "MAPE"]):
    bars = ax.barh(results_df["Model"], results_df[metric], color=palette)
    ax.set_xlabel(metric + (" (%)" if metric == "MAPE" else " (kWh)"), fontsize=11)
    ax.set_title(metric, fontsize=12); ax.invert_yaxis()
    for bar, v in zip(bars, results_df[metric]):
        ax.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2,
                f"{v:.2f}", va="center", fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "bdg2_dl_metrics_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()

# Forecast time series
n_show = 168
fig, axes = plt.subplots(len(MODEL_DEFS), 1, figsize=(14, 3.5*len(MODEL_DEFS)), sharex=True)
fig.suptitle("BDG-2 Day-Ahead Forecast vs Actual – First Week of Test Set", fontsize=13)
for ax, (name, (pi, ti)) in zip(axes, all_preds.items()):
    ax.plot(ti[:n_show], color="black", lw=1.5, label="Actual")
    ax.plot(pi[:n_show], color="tomato", lw=1.2, ls="--", label=name)
    ax.set_title(f"{name}  (MAE={mean_absolute_error(ti[:n_show],pi[:n_show]):.2f} kWh)", fontsize=9)
    ax.set_ylabel("kWh", fontsize=8); ax.legend(fontsize=8)
axes[-1].set_xlabel("Hours", fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "bdg2_dl_forecast_timeseries.png"), dpi=150, bbox_inches="tight")
plt.close()

print("\n" + "="*60)
print("BDG-2 DL RESULTS (sorted by RMSE)")
print("="*60)
print(results_df[["Model","MAE","RMSE","MAPE","CV_RMSE","R2","Params_M","Train_Time_s"]].to_string(index=False))
print(f"\nSaved → {OUTPUT_DIR}/bdg2_dl_metrics.csv")
print("Done.")
