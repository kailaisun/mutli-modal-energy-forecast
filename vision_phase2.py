"""
Vision-Enhanced Phase 2: DeiT-Small & DINOv2 with best config
Best config from sweep: RP encoding, heads=4, frozen backbone, 2 cross-attn layers
Compares: Baseline | DeiT-Tiny+RP | DeiT-Small+RP | DINOv2+RP | DINOv2+GAF+RP+MTF
"""

import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import timm

warnings.filterwarnings("ignore")
torch.manual_seed(42); np.random.seed(42)
torch.backends.cudnn.enabled = False   # cuDNN init issue on this system

DATA_ROOT  = "/data/buildings_bench/BuildingsBench/Electricity"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN, HORIZON, N_BUILDINGS = 168, 24, 10
BATCH_SIZE = 128
EPOCHS, PATIENCE = 40, 6
IMG_SIZE, IMG_FULL = 64, 224
print(f"Device: {DEVICE}")

# ── Encoders ──────────────────────────────────────────────────────────────────
def encode_gaf(x, size=IMG_SIZE):
    if len(x) > size: x = x[np.linspace(0,len(x)-1,size).astype(int)]
    mn,mx = x.min(),x.max()
    if mx-mn<1e-8: return np.zeros((size,size),np.float32)
    x = np.clip(2*(x-mn)/(mx-mn)-1,-1,1)
    phi = np.arccos(x)
    return np.cos(phi[:,None]+phi[None,:]).astype(np.float32)

def encode_rp(x, size=IMG_SIZE, thr=0.1):
    if len(x) > size: x = x[np.linspace(0,len(x)-1,size).astype(int)]
    mn,mx = x.min(),x.max()
    if mx-mn<1e-8: return np.zeros((size,size),np.float32)
    x = (x-mn)/(mx-mn)
    d = np.abs(x[:,None]-x[None,:])
    return (d<=np.percentile(d,thr*100)).astype(np.float32)

def encode_mtf(x, size=IMG_SIZE, n_bins=8):
    if len(x) > size: x = x[np.linspace(0,len(x)-1,size).astype(int)]
    q = np.percentile(x,np.linspace(0,100,n_bins+1)); q[-1]+=1e-8
    bins = np.digitize(x,q[1:-1])
    M = np.zeros((n_bins,n_bins),np.float32)
    for i in range(len(bins)-1): M[bins[i],bins[i+1]]+=1
    rs = M.sum(1,keepdims=True); M = np.where(rs>0,M/rs,1./n_bins)
    return M[bins[:,None],bins[None,:]].astype(np.float32)

ENCODERS = {"GAF":encode_gaf,"RP":encode_rp,"MTF":encode_mtf}

def to_rgb(img):
    mn,mx = img.min(),img.max()
    if mx-mn>1e-8: img=(img-mn)/(mx-mn)
    rgb = np.stack([img,img,img],0)
    mean = np.array([0.485,0.456,0.406],dtype=np.float32)[:,None,None]
    std  = np.array([0.229,0.224,0.225],dtype=np.float32)[:,None,None]
    return (rgb-mean)/std

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading data...")
dfs = [pd.read_csv(os.path.join(DATA_ROOT,f"LD2011_2014_clean={y}.csv"),
       parse_dates=["timestamp"],index_col="timestamp") for y in [2011,2012,2013,2014]]
data = pd.concat(dfs).sort_index().iloc[:,:N_BUILDINGS].dropna()
buildings = list(data.columns); cutoff = pd.Timestamp("2014-01-01")
scalers,train_sc,test_sc = {},{},{}
for bld in buildings:
    sc = StandardScaler()
    train_sc[bld] = sc.fit_transform(data.loc[data.index<cutoff,bld].values.reshape(-1,1)).ravel()
    test_sc[bld]  = sc.transform(data.loc[data.index>=cutoff,bld].values.reshape(-1,1)).ravel()
    scalers[bld]  = sc

class VDS(Dataset):
    def __init__(self, sd, enc_names):
        self.enc_names=enc_names; self.bld_list=list(sd.keys()); self.s=[]
        for bi,bld in enumerate(self.bld_list):
            v=sd[bld].astype(np.float32)
            for i in range(SEQ_LEN,len(v)-HORIZON): self.s.append((v[i-SEQ_LEN:i],v[i+HORIZON-1],bi))
    def __len__(self): return len(self.s)
    def __getitem__(self,idx):
        x,y,b=self.s[idx]
        imgs={n:torch.from_numpy(to_rgb(ENCODERS[n](x,IMG_SIZE))) for n in self.enc_names}
        return torch.from_numpy(x).unsqueeze(-1),imgs,torch.tensor(y,dtype=torch.float32),b

def loaders(enc_names):
    tr=VDS(train_sc,enc_names); te=VDS(test_sc,enc_names)
    kw=dict(num_workers=4,pin_memory=True)
    return DataLoader(tr,BATCH_SIZE,shuffle=True,**kw),DataLoader(te,BATCH_SIZE,**kw),tr,te

def inv(pn,bids):
    out=np.empty_like(pn)
    for bi,bld in enumerate(buildings):
        m=bids==bi
        if m.any(): out[m]=scalers[bld].inverse_transform(pn[m].reshape(-1,1)).ravel()
    return out

# ── Backbones ─────────────────────────────────────────────────────────────────
class DeiTBB(nn.Module):
    def __init__(self, size="tiny", freeze=True):
        super().__init__()
        self.vit = timm.create_model(f"deit_{size}_patch16_224",pretrained=True,num_classes=0)
        self.d_model = self.vit.embed_dim
        if freeze:
            for p in self.vit.parameters(): p.requires_grad_(False)
    def forward(self,x):
        x=F.interpolate(x,(IMG_FULL,IMG_FULL),mode="bilinear",align_corners=False)
        return self.vit.forward_features(x)[:,1:]   # drop CLS (B,196,d)

class DINOv2BB(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        self.vit = torch.hub.load("facebookresearch/dinov2","dinov2_vits14",pretrained=True,verbose=False)
        self.d_model = 384
        if freeze:
            for p in self.vit.parameters(): p.requires_grad_(False)
    def forward(self,x):
        x=F.interpolate(x,(IMG_FULL,IMG_FULL),mode="bilinear",align_corners=False)
        return self.vit.forward_features(x)["x_norm_patchtokens"]  # (B,256,384)

# ── PatchTST encoder (returns all patch tokens) ───────────────────────────────
class PatchTSTEnc(nn.Module):
    def __init__(self,d=64,nhead=4,nlayers=3,dff=256,drop=0.1):
        super().__init__()
        self.pl,self.stride=24,12
        np_=(SEQ_LEN-self.pl)//self.stride+1
        self.proj=nn.Linear(self.pl,d); self.pos=nn.Parameter(torch.zeros(1,np_,d))
        enc=nn.TransformerEncoderLayer(d,nhead,dff,drop,batch_first=True)
        self.enc=nn.TransformerEncoder(enc,nlayers); self.d=d; self.np=np_
    def forward(self,x):
        x=x.squeeze(-1).unfold(1,self.pl,self.stride)
        return self.enc(self.proj(x)+self.pos)   # (B,np,d)

# ── Cross-Attention Fusion ────────────────────────────────────────────────────
class XAttnFusion(nn.Module):
    def __init__(self,d_ts,d_img,d=128,heads=4,drop=0.1,nlayers=2):
        super().__init__()
        self.qp=nn.Linear(d_ts,d); self.kvp=nn.Linear(d_img,d)
        self.layers=nn.ModuleList([nn.ModuleDict({
            "attn":nn.MultiheadAttention(d,heads,dropout=drop,batch_first=True),
            "n1":nn.LayerNorm(d),"n2":nn.LayerNorm(d),
            "ff":nn.Sequential(nn.Linear(d,d*4),nn.GELU(),nn.Dropout(drop),nn.Linear(d*4,d))
        }) for _ in range(nlayers)])
        self.pool=nn.AdaptiveAvgPool1d(1); self.d=d
    def forward(self,ts,img_list):
        kv=self.kvp(torch.cat(img_list,1)); q=self.qp(ts)
        for l in self.layers:
            a,_=l["attn"](q,kv,kv); q=l["n1"](q+a); q=l["n2"](q+l["ff"](q))
        return self.pool(q.transpose(1,2)).squeeze(-1)

class VisionModel(nn.Module):
    def __init__(self,bb,enc_names,d_ts=64,d_cross=128,heads=4,nlayers=2,drop=0.1):
        super().__init__()
        self.enc_names=enc_names; self.ts=PatchTSTEnc(d=d_ts)
        self.bb=bb
        self.fusion=XAttnFusion(d_ts,bb.d_model,d_cross,heads,drop,nlayers)
        self.head=nn.Sequential(nn.Linear(d_cross,128),nn.GELU(),nn.Dropout(drop),
                                nn.Linear(128,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,ts,imgs):
        t=self.ts(ts); im=[self.bb(imgs[n]) for n in self.enc_names]
        return self.head(self.fusion(t,im)).squeeze(-1)

class Baseline(nn.Module):
    def __init__(self,d=64,nhead=4,nlayers=3,dff=256,drop=0.1):
        super().__init__()
        self.enc=PatchTSTEnc(d,nhead,nlayers,dff,drop)
        self.head=nn.Sequential(nn.Linear(d*self.enc.np,256),nn.GELU(),nn.Dropout(drop),
                                nn.Linear(256,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,ts,imgs=None): return self.head(self.enc(ts).flatten(1)).squeeze(-1)

# ── Train/eval ────────────────────────────────────────────────────────────────
def run(model,enc_names,label,lr_bb=0.0,lr=1e-3):
    trl,tel,trd,ted=loaders(enc_names)
    model=model.to(DEVICE)
    bb_params=[p for n,p in model.named_parameters() if "bb." in n]
    main_params=[p for n,p in model.named_parameters() if "bb." not in n]
    opt=torch.optim.AdamW([{"params":main_params,"lr":lr},{"params":bb_params,"lr":lr_bb}],weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    crit=nn.HuberLoss(delta=1.0)
    best,bs,pat=float("inf"),None,0; tl_,vl_=[],[]
    t0=time.time()
    for ep in range(1,EPOCHS+1):
        model.train(); rl=0.
        for ts,imgs,yb,_ in trl:
            ts=ts.to(DEVICE); yb=yb.to(DEVICE)
            id_={k:v.to(DEVICE) for k,v in imgs.items()}
            opt.zero_grad(); loss=crit(model(ts,id_),yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); rl+=loss.item()*len(yb)
        tl_.append(rl/len(trd))
        model.eval(); vl=0.
        with torch.no_grad():
            for ts,imgs,yb,_ in tel:
                id_={k:v.to(DEVICE) for k,v in imgs.items()}
                vl+=crit(model(ts.to(DEVICE),id_),yb.to(DEVICE)).item()*len(yb)
        vl_.append(vl/len(ted)); sched.step()
        if ep%5==0: print(f"    ep {ep:3d}/{EPOCHS}  train={tl_[-1]:.4f}  val={vl_[-1]:.4f}")
        if vl_[-1]<best: best,pat,bs=vl_[-1],0,{k:v.cpu().clone() for k,v in model.state_dict().items()}
        else:
            pat+=1
            if pat>=PATIENCE: print(f"    Early stop ep {ep}"); break
    model.load_state_dict(bs); elapsed=time.time()-t0
    model.eval(); pn,tn,bids=[],[],[]
    with torch.no_grad():
        for ts,imgs,yb,bidx in tel:
            id_={k:v.to(DEVICE) for k,v in imgs.items()}
            pn.append(model(ts.to(DEVICE),id_).cpu().numpy())
            tn.append(yb.numpy()); bids.append(bidx.numpy())
    pn=np.concatenate(pn); tn=np.concatenate(tn); bids=np.concatenate(bids)
    pk=np.clip(inv(pn,bids),0,None); tk=inv(tn,bids)
    mae=mean_absolute_error(tk,pk); rmse=np.sqrt(mean_squared_error(tk,pk))
    r2=r2_score(tk,pk); mape=np.mean(np.abs((tk-pk)/(tk+1e-6)))*100
    cv=rmse/(tk.mean()+1e-6)*100
    np_=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  → MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.1f}%  R²={r2:.4f}  ({elapsed:.0f}s)")
    return dict(Model=label,MAE=mae,RMSE=rmse,MAPE=mape,CV_RMSE=cv,
                R2=r2,Params_M=round(np_/1e6,2),Train_Time_s=round(elapsed,1)),\
           (pk,tk),(tl_,vl_)

# ── Experiments ───────────────────────────────────────────────────────────────
# Best from sweep: RP, heads=4, frozen, 2 layers
# Also test: DINOv2 + GAF+RP+MTF (MultiView) since DINOv2 is richer
EXPS = [
    ("PatchTST Baseline",             lambda: Baseline(),                                      [],                  0.0),
    ("DeiT-Tiny  | RP | frozen",      lambda: VisionModel(DeiTBB("tiny",  True), ["RP"]),      ["RP"],              0.0),
    ("DeiT-Small | RP | frozen",      lambda: VisionModel(DeiTBB("small", True), ["RP"]),      ["RP"],              0.0),
    ("DINOv2-S14 | RP | frozen",      lambda: VisionModel(DINOv2BB(True),        ["RP"]),      ["RP"],              0.0),
    ("DINOv2-S14 | GAF+RP+MTF | frozen", lambda: VisionModel(DINOv2BB(True), ["GAF","RP","MTF"]), ["GAF","RP","MTF"], 0.0),
]

results,preds,curves=[],{},{}
for label,model_fn,enc,lr_bb in EXPS:
    model=model_fn()
    np_=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*55}\n  {label}  ({np_/1e6:.2f}M trainable params)\n{'='*55}")
    m,pr,cu=run(model,enc,label,lr_bb=lr_bb)
    results.append(m); preds[label]=pr; curves[label]=cu

df=pd.DataFrame(results).sort_values("RMSE")
base_rmse=df[df["Model"]=="PatchTST Baseline"]["RMSE"].iloc[0]
df["RMSE_Δ%"]=(base_rmse-df["RMSE"])/base_rmse*100
df.to_csv(os.path.join(OUTPUT_DIR,"vision_adv_phase2.csv"),index=False,float_format="%.4f")

print("\n"+"="*65)
print("FINAL RESULTS")
print("="*65)
print(df[["Model","MAE","RMSE","MAPE","R2","Params_M","RMSE_Δ%"]].to_string(index=False))

# ── Figures ───────────────────────────────────────────────────────────────────
# Fig 1: RMSE + improvement
fig,axes=plt.subplots(1,2,figsize=(15,6))
fig.suptitle("Advanced Vision-Enhanced PatchTST\nDeiT / DINOv2 + Cross-Attention Fusion",fontsize=13)
colors=["#e74c3c" if "Baseline" in m else "#2ecc71" for m in df["Model"]]
ax=axes[0]; bars=ax.barh(df["Model"],df["RMSE"],color=colors)
ax.set_xlabel("RMSE (kWh)",fontsize=11); ax.set_title("RMSE (lower=better)"); ax.invert_yaxis()
for bar,v in zip(bars,df["RMSE"]):
    ax.text(bar.get_width()+0.2,bar.get_y()+bar.get_height()/2,f"{v:.2f}",va="center",fontsize=9)

no_base=df[df["Model"]!="PatchTST Baseline"].sort_values("RMSE_Δ%",ascending=False)
ax=axes[1]; colors2=["#2ecc71" if v>=0 else "#e74c3c" for v in no_base["RMSE_Δ%"]]
bars=ax.barh(no_base["Model"],no_base["RMSE_Δ%"],color=colors2)
ax.axvline(0,color="black",lw=1); ax.set_xlabel("RMSE Improvement (%)",fontsize=11)
ax.set_title("vs PatchTST Baseline"); ax.invert_yaxis()
for bar,v in zip(bars,no_base["RMSE_Δ%"]):
    ax.text(max(bar.get_width(),0)+0.05,bar.get_y()+bar.get_height()/2,
            f"{v:+.2f}%",va="center",fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR,"vadv_phase2_final.png"),dpi=150,bbox_inches="tight")
plt.close()

# Fig 2: Full evolution (no-vision → simple CNN → cross-attn ViT)
evol={
    "PatchTST\n(no vision)":           83.021,
    "PatchTST+CNN+concat\n(best prev)": 80.820,  # PatchTST+RP from vision_enhanced
}
for _,row in df[df["Model"]!="PatchTST Baseline"].iterrows():
    evol[row["Model"].replace(" | ","\n")] = row["RMSE"]
items=sorted(evol.items(),key=lambda x:x[1],reverse=True)
labels,vals=zip(*items)
colors_e=["#e74c3c","#f39c12"]+["#2ecc71"]*len(df[df["Model"]!="PatchTST Baseline"])
fig,ax=plt.subplots(figsize=(13,6))
bars=ax.barh(labels,vals,color=colors_e[:len(vals)])
ax.set_xlabel("RMSE (kWh)",fontsize=11)
ax.set_title("RMSE Evolution: No Vision → Simple CNN → Cross-Attention ViT Backbone",fontsize=12)
ax.invert_yaxis()
for bar,v in zip(bars,vals):
    ax.text(bar.get_width()+0.1,bar.get_y()+bar.get_height()/2,f"{v:.2f}",va="center",fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR,"vadv_evolution.png"),dpi=150,bbox_inches="tight")
plt.close()

# Fig 3: Loss curves
fig,axes=plt.subplots(1,len(curves),figsize=(5*len(curves),4))
fig.suptitle("Training Curves",fontsize=12)
for ax,(name,(tl,vl)) in zip(axes,curves.items()):
    ax.plot(tl,label="Train"); ax.plot(vl,label="Val")
    ax.set_title(name[:35],fontsize=8); ax.legend(fontsize=7)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR,"vadv_loss_curves.png"),dpi=150,bbox_inches="tight")
plt.close()

# Fig 4: Heatmap
norm=df.set_index("Model")[["MAE","RMSE","MAPE","CV_RMSE","R2"]].copy()
for c in ["MAE","RMSE","MAPE","CV_RMSE"]:
    mn,mx=norm[c].min(),norm[c].max(); norm[c]=1-(norm[c]-mn)/(mx-mn+1e-9)
mn,mx=norm["R2"].min(),norm["R2"].max(); norm["R2"]=(norm["R2"]-mn)/(mx-mn+1e-9)
fig,ax=plt.subplots(figsize=(12,5))
sns.heatmap(norm.T,annot=True,fmt=".3f",cmap="YlGn",linewidths=0.4,ax=ax,vmin=0,vmax=1)
ax.set_title("Normalised Performance Heatmap – Phase 2 Models",fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR,"vadv_heatmap.png"),dpi=150,bbox_inches="tight")
plt.close()

# Fig 5: sweep summary (load from sweep csv if exists)
sweep_path=os.path.join(OUTPUT_DIR,"vision_adv_sweep.csv")
if os.path.exists(sweep_path):
    sw=pd.read_csv(sweep_path)
    no_base_sw=sw[sw["Model"]!="PatchTST Baseline"].sort_values("RMSE_improvement_%",ascending=False)
    fig,ax=plt.subplots(figsize=(12,6))
    colors_sw=["#2ecc71" if v>=0 else "#e74c3c" for v in no_base_sw["RMSE_improvement_%"]]
    bars=ax.barh(no_base_sw["Model"],no_base_sw["RMSE_improvement_%"],color=colors_sw)
    ax.axvline(0,color="black",lw=1)
    ax.set_xlabel("RMSE Improvement (%)",fontsize=11)
    ax.set_title("Phase 1 Hyperparameter Sweep Results (DeiT-Tiny)",fontsize=12)
    ax.invert_yaxis()
    for bar,v in zip(bars,no_base_sw["RMSE_improvement_%"]):
        ax.text(max(bar.get_width(),0)+0.05,bar.get_y()+bar.get_height()/2,
                f"{v:+.2f}%",va="center",fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR,"vadv_sweep_improvement.png"),dpi=150,bbox_inches="tight")
    plt.close()

print("\nAll done.")
