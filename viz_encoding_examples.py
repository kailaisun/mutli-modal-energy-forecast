"""
Generate a publication-quality figure showing raw time series → 4 image encodings.
Shows one week (168h) of building energy consumption and its GAF / RP / MTF / Spectrogram.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import stft

DATA_ROOT  = "/data/buildings_bench/BuildingsBench/Electricity"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load one week of data ──────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA_ROOT, "LD2011_2014_clean=2013.csv"),
                 parse_dates=["timestamp"], index_col="timestamp")
# Pick building MT_168 (medium-size, clear daily pattern) and a week in June 2013
col = df.columns[7]
week = df[col].loc["2013-06-03":"2013-06-09 23:00"].values[:168].astype(np.float64)

# ── Encoding functions ─────────────────────────────────────────────────────────
SIZE = 64

def encode_gaf(x, size=SIZE):
    if len(x) > size: x = x[np.linspace(0, len(x)-1, size).astype(int)]
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8: return np.zeros((size, size))
    x = np.clip(2*(x - mn)/(mx - mn) - 1, -1, 1)
    phi = np.arccos(x)
    return np.cos(phi[:, None] + phi[None, :])

def encode_rp(x, size=SIZE, thr=0.1):
    if len(x) > size: x = x[np.linspace(0, len(x)-1, size).astype(int)]
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8: return np.zeros((size, size))
    x = (x - mn)/(mx - mn)
    d = np.abs(x[:, None] - x[None, :])
    return (d <= np.percentile(d, thr * 100)).astype(float)

def encode_mtf(x, size=SIZE, n_bins=8):
    if len(x) > size: x = x[np.linspace(0, len(x)-1, size).astype(int)]
    q = np.percentile(x, np.linspace(0, 100, n_bins+1)); q[-1] += 1e-8
    bins = np.digitize(x, q[1:-1])
    M = np.zeros((n_bins, n_bins))
    for i in range(len(bins)-1): M[bins[i], bins[i+1]] += 1
    rs = M.sum(1, keepdims=True); M = np.where(rs > 0, M/rs, 1./n_bins)
    return M[bins[:, None], bins[None, :]]

def encode_spec(x):
    f, t, Zxx = stft(x, nperseg=24, noverlap=20)
    mag = np.abs(Zxx)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    return mag, f, t

# ── Compute encodings ──────────────────────────────────────────────────────────
gaf  = encode_gaf(week)
rp   = encode_rp(week)
mtf  = encode_mtf(week)
spec, freq, t_spec = encode_spec(week)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(2, 5, figure=fig, hspace=0.45, wspace=0.35)

# Row 0: Time series (wide) + GAF + RP
# Row 1: MTF + Spectrogram + description table

# ── Time series (spans top-left 2 columns) ────────────────────────────────────
ax_ts = fig.add_subplot(gs[0, :2])
hours = np.arange(168)
ax_ts.plot(hours, week, color="#1565C0", linewidth=1.4)
ax_ts.fill_between(hours, week, alpha=0.15, color="#1565C0")
for d in range(7):
    ax_ts.axvline(d*24, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)
ax_ts.set_xticks(np.arange(0, 169, 24))
ax_ts.set_xticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun","Mon"], fontsize=9)
ax_ts.set_xlabel("Time", fontsize=10)
ax_ts.set_ylabel("Load (kWh)", fontsize=10)
ax_ts.set_title(f"Raw Time Series — 1 Week (168h)\n{col}", fontsize=11, fontweight="bold")
ax_ts.grid(axis="y", alpha=0.3)

# ── GAF ───────────────────────────────────────────────────────────────────────
ax_gaf = fig.add_subplot(gs[0, 2])
im = ax_gaf.imshow(gaf, cmap="coolwarm", aspect="auto", origin="lower")
ax_gaf.set_title("GAF\n(Gramian Angular Field)", fontsize=10, fontweight="bold")
ax_gaf.set_xlabel("Time step"); ax_gaf.set_ylabel("Time step")
plt.colorbar(im, ax=ax_gaf, shrink=0.8, label="cos(φᵢ+φⱼ)")
# Annotate diagonal = same time point
ax_gaf.plot([0, SIZE-1], [0, SIZE-1], color="white", linewidth=0.8, linestyle="--", alpha=0.6)

# ── RP ────────────────────────────────────────────────────────────────────────
ax_rp = fig.add_subplot(gs[0, 3])
ax_rp.imshow(rp, cmap="binary", aspect="auto", origin="lower")
ax_rp.set_title("RP\n(Recurrence Plot)", fontsize=10, fontweight="bold")
ax_rp.set_xlabel("Time step"); ax_rp.set_ylabel("Time step")
# Highlight daily diagonals
for offset in [24, 48, 72, 96, 120, 144]:
    scaled = int(offset * SIZE / 168)
    ax_rp.plot([0, SIZE-scaled-1], [scaled, SIZE-1], color="red", linewidth=0.8, alpha=0.5)

# ── Spectrogram ───────────────────────────────────────────────────────────────
ax_sp = fig.add_subplot(gs[0, 4])
im2 = ax_sp.pcolormesh(t_spec, freq, spec, cmap="inferno", shading="auto")
ax_sp.set_title("Spectrogram\n(STFT Magnitude)", fontsize=10, fontweight="bold")
ax_sp.set_xlabel("Time step"); ax_sp.set_ylabel("Frequency (cycles/h)")
plt.colorbar(im2, ax=ax_sp, shrink=0.8, label="Magnitude")
# Mark 24h period
if len(freq) > 1:
    f24 = 1/24
    ax_sp.axhline(f24, color="cyan", linewidth=1.2, linestyle="--", alpha=0.8, label="24h")
    ax_sp.legend(fontsize=8, loc="upper right")

# ── MTF (row 1, col 0-1) ──────────────────────────────────────────────────────
ax_mtf = fig.add_subplot(gs[1, :2])
im3 = ax_mtf.imshow(mtf, cmap="viridis", aspect="auto", origin="lower")
ax_mtf.set_title("MTF (Markov Transition Field)\nTransition probability between quantile bins over time",
                 fontsize=10, fontweight="bold")
ax_mtf.set_xlabel("Time step"); ax_mtf.set_ylabel("Time step")
plt.colorbar(im3, ax=ax_mtf, shrink=0.8, label="Transition prob")

# ── Description table (row 1, col 2-4) ───────────────────────────────────────
ax_desc = fig.add_subplot(gs[1, 2:])
ax_desc.axis("off")

table_data = [
    ["Encoding", "Captures", "Key Property", "Phase 2 RMSE Δ%"],
    ["GAF",  "Temporal correlations\n(pairwise sum of angles)", "Outer product of arccos-\ntransformed values", "+4.03% (DINOv2)"],
    ["RP",   "Periodicity &\nregime changes", "Binary pairwise\ndistance matrix",    "★ +4.64% (DeiT-Tiny)"],
    ["MTF",  "State transition\ndynamics", "Markov transition\nprobabilities",        "+3.38% (DINOv2 multi)"],
    ["Spec", "Frequency patterns\nover time", "STFT magnitude\nspectrum",             "−1.27% ↓ (hurts)"],
]

col_widths = [0.12, 0.28, 0.30, 0.28]
row_colors_list = [["#1565C0"]*4, ["#F3F3F3"]*4, ["#E8F5E9"]*4,
                   ["#E8F5E9"]*4, ["#FFEBEE"]*4]
text_colors = [["white"]*4, ["black"]*4, ["black"]*4, ["black"]*4, ["black"]*4]

t = ax_desc.table(
    cellText=table_data,
    loc="center",
    cellLoc="center",
    bbox=[0.0, 0.0, 1.0, 1.0]
)
t.auto_set_font_size(False); t.set_fontsize(9)
for (row, col), cell in t.get_celld().items():
    if row == 0:
        cell.set_facecolor("#1565C0"); cell.set_text_props(color="white", fontweight="bold")
    elif row == 4:
        cell.set_facecolor("#FFEBEE")
    elif row % 2 == 0:
        cell.set_facecolor("#F5F5F5")
    cell.set_edgecolor("white"); cell.set_linewidth(0.5)

ax_desc.set_title("Image Encoding Summary", fontsize=10, fontweight="bold", pad=10)

# ── Overall title ─────────────────────────────────────────────────────────────
fig.suptitle("Time Series → 2D Image Encoding for Vision-Enhanced Forecasting",
             fontsize=14, fontweight="bold", y=0.98)

fig.savefig(os.path.join(OUTPUT_DIR, "ts_encoding_examples.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved results/ts_encoding_examples.png")
