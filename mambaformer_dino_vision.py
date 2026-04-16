"""
MambaFormer + Frozen DINOv2 Vision via Cross-Attention.

Architecture:
  - MambaFormerBackbone (nlayers=2) → last token → Q (B, 1, d_model)
  - Frozen DINOv2-small (ViT-S/14) → 256 patch tokens (B, 256, 384)
  - Learnable KV projection (2-layer MLP): 384 → d_model → K, V
  - MultiHeadCrossAttention(Q=TS, K=img, V=img) → (B, d_model)
  - Head → prediction

For MultiView: each encoding goes through DINOv2 separately,
K/V tokens are concatenated (4×256 = 1024 tokens).
"""

import os, time, warnings
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
torch.manual_seed(42); np.random.seed(42)
torch.backends.cudnn.enabled = False

DATA_ROOT  = "/data/buildings_bench/BuildingsBench/Electricity"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN, HORIZON, N_BUILDINGS = 168, 24, 10
BATCH_SIZE, EPOCHS, PATIENCE, LR = 256, 20, 5, 1e-3
IMG_SIZE   = 32
D_MODEL    = 64
print(f"Device: {DEVICE}")

# ── Image encoders ─────────────────────────────────────────────────────────────
def encode_gaf(x, size=IMG_SIZE):
    if len(x) > size: x = x[np.linspace(0, len(x)-1, size).astype(int)]
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8: return np.zeros((size, size), np.float32)
    x = np.clip(2*(x-mn)/(mx-mn)-1, -1, 1); phi = np.arccos(x)
    return np.cos(phi[:,None] + phi[None,:]).astype(np.float32)

def encode_rp(x, size=IMG_SIZE, thr=0.1):
    if len(x) > size: x = x[np.linspace(0, len(x)-1, size).astype(int)]
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8: return np.zeros((size, size), np.float32)
    x = (x-mn)/(mx-mn); d = np.abs(x[:,None]-x[None,:])
    return (d <= np.percentile(d, thr*100)).astype(np.float32)

def encode_mtf(x, size=IMG_SIZE, n_bins=8):
    if len(x) > size: x = x[np.linspace(0, len(x)-1, size).astype(int)]
    q = np.percentile(x, np.linspace(0, 100, n_bins+1)); q[-1] += 1e-8
    bins = np.digitize(x, q[1:-1])
    M = np.zeros((n_bins, n_bins))
    for i in range(len(bins)-1): M[bins[i], bins[i+1]] += 1
    rs = M.sum(1, keepdims=True); M = np.where(rs > 0, M/rs, 1./n_bins)
    return M[bins[:,None], bins[None,:]].astype(np.float32)

def encode_spec(x, size=IMG_SIZE):
    n = len(x); nf = min(32, n//4); hop = max(1, (n-nf)//(size-1))
    frames = []
    for s in range(0, n-nf+1, hop):
        frames.append(np.abs(np.fft.rfft(x[s:s+nf]*np.hanning(nf))))
        if len(frames) >= size: break
    while len(frames) < size: frames.append(frames[-1] if frames else np.zeros(nf//2+1))
    s2d = np.stack(frames[:size], 1)[np.linspace(0, len(frames[0])-1, size).astype(int), :]
    s2d = np.log1p(s2d); mn, mx = s2d.min(), s2d.max()
    if mx - mn > 1e-8: s2d = (s2d-mn)/(mx-mn)
    return s2d.astype(np.float32)

ENCODERS = {"GAF": encode_gaf, "RP": encode_rp, "MTF": encode_mtf, "Spec": encode_spec}

# ── Data ───────────────────────────────────────────────────────────────────────
print("Loading data...")
dfs = [pd.read_csv(os.path.join(DATA_ROOT, f"LD2011_2014_clean={y}.csv"),
       parse_dates=["timestamp"], index_col="timestamp") for y in [2011, 2012, 2013, 2014]]
data = pd.concat(dfs).sort_index()
buildings = data.columns[:N_BUILDINGS].tolist()
data = data[buildings].dropna()
cutoff = pd.Timestamp("2014-01-01")

scalers, train_sc, test_sc = {}, {}, {}
for bld in buildings:
    sc = StandardScaler()
    train_sc[bld] = sc.fit_transform(data.loc[data.index < cutoff, bld].values.reshape(-1,1)).ravel()
    test_sc[bld]  = sc.transform(data.loc[data.index >= cutoff, bld].values.reshape(-1,1)).ravel()
    scalers[bld]  = sc

class VDS(Dataset):
    def __init__(self, sd, enc_names):
        self.enc_names = enc_names; self.samples = []
        for bi, bld in enumerate(buildings):
            v = sd[bld].astype(np.float32)
            for i in range(SEQ_LEN, len(v)-HORIZON):
                self.samples.append((v[i-SEQ_LEN:i], v[i+HORIZON-1], bi))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, y, b = self.samples[idx]
        imgs = {n: torch.from_numpy(ENCODERS[n](x, IMG_SIZE)).unsqueeze(0) for n in self.enc_names}
        return torch.from_numpy(x).unsqueeze(-1), imgs, torch.tensor(y, dtype=torch.float32), b

def make_loaders(enc_names):
    kw = dict(num_workers=4, pin_memory=True)
    return (DataLoader(VDS(train_sc, enc_names), BATCH_SIZE, shuffle=True,  **kw),
            DataLoader(VDS(test_sc,  enc_names), BATCH_SIZE, shuffle=False, **kw))

def inv(p, bids):
    out = np.empty_like(p)
    for bi, bld in enumerate(buildings):
        m = bids == bi
        if m.any(): out[m] = scalers[bld].inverse_transform(p[m].reshape(-1,1)).ravel()
    return out

# ── MambaFormer backbone ───────────────────────────────────────────────────────
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        D = d_model * expand; self.D_inner = D; self.d_state = d_state
        self.in_proj  = nn.Linear(d_model, D*2)
        self.conv1d   = nn.Conv1d(D, D, d_conv, padding=d_conv-1, groups=D)
        self.x_proj   = nn.Linear(D, d_state*2+D)
        self.dt_proj  = nn.Linear(D, D, bias=True)
        self.A_log    = nn.Parameter(torch.log(torch.arange(1, d_state+1, dtype=torch.float)
                                               .unsqueeze(0).expand(D, -1)))
        self.D_skip   = nn.Parameter(torch.ones(D))
        self.out_proj = nn.Linear(D, d_model)
        self.norm = nn.LayerNorm(d_model); self.drop = nn.Dropout(dropout)

    @staticmethod
    def _pscan(a, b):
        s = torch.exp(torch.cumsum(torch.log(torch.clamp(a, 1e-6)), 1))
        return s * torch.cumsum(b / (s + 1e-8), 1)

    def forward(self, x):
        B, T, _ = x.shape; res = x
        xz = self.in_proj(x); x_, z = xz.chunk(2, -1)
        x_ = self.conv1d(x_.permute(0,2,1))[:,:,:T].permute(0,2,1); x_ = F.silu(x_)
        bcd = self.x_proj(x_)
        B_s = bcd[:,:,:self.d_state]; C_s = bcd[:,:,self.d_state:2*self.d_state]
        dt  = F.softplus(self.dt_proj(bcd[:,:,2*self.d_state:]))
        A   = -torch.exp(self.A_log); dA = torch.exp(dt.unsqueeze(-1)*A)
        dB  = dt.unsqueeze(-1) * B_s.unsqueeze(2).expand(-1,-1,self.D_inner,-1)
        h   = self._pscan(dA, dB * x_.unsqueeze(-1).expand(-1,-1,-1,self.d_state))
        y   = (h * C_s.unsqueeze(2)).sum(-1) + self.D_skip * x_
        return self.norm(self.drop(self.out_proj(y * F.silu(z))) + res)

class MambaFormerBackbone(nn.Module):
    """nlayers=2, returns full sequence (B, T, d_model) for cross-attention."""
    def __init__(self, d_model=D_MODEL, d_state=16, nhead=4, nlayers=2, dim_ff=256, dropout=0.1):
        super().__init__()
        self.proj   = nn.Linear(1, d_model)
        self.pos    = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        self.layers = nn.ModuleList()
        for _ in range(nlayers):
            self.layers.append(MambaBlock(d_model, d_state=d_state, dropout=dropout))
            self.layers.append(nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True))
        self.norm    = nn.LayerNorm(d_model)
        self.out_dim = d_model

    def forward(self, x):
        x = self.proj(x) + self.pos
        for layer in self.layers: x = layer(x)
        return self.norm(x)  # (B, T, d_model) — full sequence

# ── Frozen DINOv2 + KV projection ─────────────────────────────────────────────
class DINOv2KVEncoder(nn.Module):
    """
    Frozen DINOv2-small → 256 patch tokens (B, 256, 384).
    Learnable 2-layer MLP projects to K and V in d_model space.
    """
    DINO_DIM = 384  # ViT-S/14

    def __init__(self, d_model=D_MODEL, n_kv_layers=2):
        super().__init__()
        print("  Loading DINOv2-small (frozen)...")
        self.dino = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vits14',
            pretrained=True, verbose=False)
        for p in self.dino.parameters():
            p.requires_grad = False
        self.dino.eval()

        def make_proj():
            layers = []
            in_d = self.DINO_DIM
            for _ in range(n_kv_layers - 1):
                layers += [nn.Linear(in_d, in_d), nn.GELU()]
            layers.append(nn.Linear(in_d, d_model))
            return nn.Sequential(*layers)

        self.k_proj = make_proj()
        self.v_proj = make_proj()

        # ImageNet normalisation constants (on device)
        self.register_buffer('img_mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('img_std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x):
        """x: (B, 1, H, W) float32 in [0,1] roughly."""
        B = x.shape[0]
        # Grayscale → 3-ch, upsample to 224×224
        x3 = x.repeat(1, 3, 1, 1)
        x3 = F.interpolate(x3, size=(224, 224), mode='bilinear', align_corners=False)
        x3 = (x3 - self.img_mean) / self.img_std

        with torch.no_grad():
            feats = self.dino.forward_features(x3)
        # DINOv2 returns dict; patch tokens at 'x_norm_patchtokens'
        patch_tokens = feats['x_norm_patchtokens']  # (B, 256, 384)

        k = self.k_proj(patch_tokens)  # (B, 256, d_model)
        v = self.v_proj(patch_tokens)  # (B, 256, d_model)
        return k, v

# ── Cross-attention fusion block ───────────────────────────────────────────────
class CrossAttnBlock(nn.Module):
    """Q from TS last token, K/V from image patch tokens."""
    def __init__(self, d_model=D_MODEL, nhead=4, dropout=0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(d_model*4, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, q, k, v):
        """q: (B, 1, d_model), k/v: (B, N_kv, d_model) → (B, 1, d_model)"""
        attn_out, _ = self.attn(q, k, v)
        q = self.norm1(q + self.drop(attn_out))
        q = self.norm2(q + self.drop(self.ff(q)))
        return q

# ── Full models ────────────────────────────────────────────────────────────────
class BaselineModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(backbone.out_dim, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, ts, imgs=None):
        return self.head(self.backbone(ts)[:, -1]).squeeze(-1)

class DINOCrossAttnModel(nn.Module):
    """
    MambaFormer Q × DINOv2 KV cross-attention vision model.
    - Q: last TS token (B, 1, d_model)
    - K, V: DINOv2 patch tokens projected per encoding, then concatenated
    - CrossAttnBlock → attended feat → head
    """
    def __init__(self, backbone, enc_names, d_model=D_MODEL, nhead=4,
                 n_kv_layers=2, n_xattn=1, dropout=0.1):
        super().__init__()
        self.enc_names = enc_names
        self.backbone  = backbone
        self.dino_enc  = DINOv2KVEncoder(d_model=d_model, n_kv_layers=n_kv_layers)
        self.q_proj    = nn.Linear(d_model, d_model)
        self.xattn_layers = nn.ModuleList(
            [CrossAttnBlock(d_model, nhead, dropout) for _ in range(n_xattn)])
        # Gate: blend attended image feat with original TS feat
        self.gate = nn.Sequential(nn.Linear(d_model*2, d_model), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64),     nn.GELU(), nn.Linear(64, 1))

    def forward(self, ts, imgs):
        ts_seq  = self.backbone(ts)           # (B, T, d_model)
        ts_last = ts_seq[:, -1]               # (B, d_model)
        q = self.q_proj(ts_last).unsqueeze(1) # (B, 1, d_model)

        # Collect K, V across encodings (concatenate along token dim)
        all_k, all_v = [], []
        for n in self.enc_names:
            k, v = self.dino_enc(imgs[n])
            all_k.append(k); all_v.append(v)
        k = torch.cat(all_k, dim=1)  # (B, N_enc*256, d_model)
        v = torch.cat(all_v, dim=1)

        # Stacked cross-attention layers
        for layer in self.xattn_layers:
            q = layer(q, k, v)
        img_feat = q.squeeze(1)  # (B, d_model)

        # Gated fusion with original TS feature
        gate = self.gate(torch.cat([ts_last, img_feat], dim=-1))
        fused = gate * img_feat + (1 - gate) * ts_last  # (B, d_model)

        return self.head(fused).squeeze(-1)

def nparams(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ── Training ───────────────────────────────────────────────────────────────────
def run(name, model, enc_names):
    tr_ld, te_ld = make_loaders(enc_names)
    model = model.to(DEVICE)
    opt   = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    crit  = nn.HuberLoss(delta=1.0)
    best, pat, bst = 1e9, 0, None; t0 = time.time()
    np_ = nparams(model)
    print(f"\n{'='*60}\n  {name}  ({np_/1e6:.2f}M trainable)\n{'='*60}")

    for ep in range(1, EPOCHS+1):
        model.train(); tl = 0
        for ts, imgs, y, _ in tr_ld:
            ts = ts.to(DEVICE); y = y.to(DEVICE)
            imgs_d = {k: v.to(DEVICE) for k, v in imgs.items()}
            opt.zero_grad()
            loss = crit(model(ts, imgs_d), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += loss.item() * len(y)

        model.eval(); vl = 0
        with torch.no_grad():
            for ts, imgs, y, _ in te_ld:
                imgs_d = {k: v.to(DEVICE) for k, v in imgs.items()}
                vl += crit(model(ts.to(DEVICE), imgs_d), y.to(DEVICE)).item() * len(y)
        tl /= len(tr_ld.dataset); vl /= len(te_ld.dataset); sched.step()
        print(f"    ep {ep:3d}/{EPOCHS}  train={tl:.4f}  val={vl:.4f}", flush=True)
        if vl < best:
            best = vl; pat = 0
            bst = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= PATIENCE: print(f"    Early stop ep {ep}"); break

    model.load_state_dict(bst); model.eval()
    ps, ts_all, bs = [], [], []
    with torch.no_grad():
        for x, imgs, y, b in te_ld:
            imgs_d = {k: v.to(DEVICE) for k, v in imgs.items()}
            ps.append(model(x.to(DEVICE), imgs_d).cpu().numpy())
            ts_all.append(y.numpy()); bs.append(b.numpy())
    ps = np.concatenate(ps); ts_all = np.concatenate(ts_all); bs = np.concatenate(bs)
    pi = np.clip(inv(ps, bs), 0, None); ti = inv(ts_all, bs)
    mae  = mean_absolute_error(ti, pi)
    rmse = np.sqrt(mean_squared_error(ti, pi))
    mape = np.mean(np.abs((ti-pi)/(ti+1e-6))) * 100
    r2   = r2_score(ti, pi)
    elapsed = time.time() - t0
    print(f"  → MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.1f}%  R²={r2:.4f}  ({elapsed:.0f}s)")
    return dict(Model=name, MAE=round(mae,4), RMSE=round(rmse,4),
                MAPE=round(mape,4), R2=round(r2,6),
                Params_M=round(np_/1e6,2), Train_s=round(elapsed,1))

# ── Experiment configs ─────────────────────────────────────────────────────────
ENC_SETS = {
    "MambaFormer-baseline":  [],
    "MambaFormer+DINO-GAF":  ["GAF"],
    "MambaFormer+DINO-RP":   ["RP"],
    "MambaFormer+DINO-MTF":  ["MTF"],
    "MambaFormer+DINO-Spec": ["Spec"],
    "MambaFormer+DINO-MV":   ["GAF", "RP", "MTF", "Spec"],
}

results = []
for name, enc_names in ENC_SETS.items():
    backbone = MambaFormerBackbone(nlayers=2)
    if not enc_names:
        model = BaselineModel(backbone)
    else:
        model = DINOCrossAttnModel(backbone, enc_names, n_kv_layers=2, n_xattn=2)
    results.append(run(name, model, enc_names))

df = pd.DataFrame(results)
base_rmse = df[df["Model"] == "MambaFormer-baseline"]["RMSE"].iloc[0]
df["RMSE_Δ%"] = ((base_rmse - df["RMSE"]) / base_rmse * 100).round(2)

out_path = os.path.join(OUTPUT_DIR, "mambaformer_dino_vision.csv")
df.to_csv(out_path, index=False)

print("\n" + "="*60 + "\nFINAL RESULTS\n" + "="*60)
print(df[["Model", "MAE", "RMSE", "MAPE", "R2", "RMSE_Δ%"]].to_string(index=False))
print(f"\nSaved → {out_path}")
print("Done.")
