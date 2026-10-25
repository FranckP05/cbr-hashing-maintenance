import numpy as np, sys, time, gc
sys.path.insert(0, "/teamspace/studios/this_studio/hecbr_sota_camelyon")
import torch
from data.loader import load_slides
from data.splitter import split_slides, build_split_matrices
from config.seeds import SPLIT_SEED
from config.paths import HECBR_MODEL_PATH
from scripts.train_hecbr_v4 import HashNetV3
from indexes.hecbr_index import HeCBRMultiProbeIndex
from indexes.linear_scan import LinearScan

device="cuda" if torch.cuda.is_available() else "cpu"
ck=torch.load(HECBR_MODEL_PATH,map_location="cpu")
model=HashNetV3(512,512,64); model.load_state_dict(ck["state_dict"]); model=model.eval().to(device)
sd,sl=load_slides(); sids=sorted(sl.keys()); b0,bv,bt=split_slides(sids,sl,seed=SPLIT_SEED)
Xb,yb,_,_,Xt,yt=build_split_matrices(base_ids=b0+bv,val_ids=[],test_ids=bt,slide_data=sd,slide_labels=sl,normalize_l2=True)
mu=Xb.mean(0); sd_=Xb.std(0)+1e-6
def enc(X,bs=65536):
    o=[]
    for i in range(0,len(X),bs):
        t=torch.tensor((X[i:i+bs]-mu)/sd_,dtype=torch.float32).to(device)
        with torch.no_grad(): o.append((model(t)>=0).cpu().numpy()); del t
    return np.concatenate(o)

rng=np.random.default_rng(0)
N=1000000; Nnew=10000
idx=rng.choice(len(Xb),N,replace=False)
base=Xb[idx]; newX=Xb[rng.choice(len(Xb),Nnew,replace=False)]
Q=Xb[rng.choice(len(Xb),50,replace=False)]
def lat(f,steps=10):
    f(Q[0]); tt=[time.perf_counter() for _ in [0]]
    out=[]
    for i in range(0,steps):
        t0=time.perf_counter(); f(Q[i]); out.append(time.perf_counter()-t0)
    return float(np.median(out))*1000

# BASELINE avant / apres ajout
sc=LinearScan(); sc.build(base)
la_before=lat(lambda q: sc.query(q,10))
t0=time.perf_counter(); sc.build(np.concatenate([base,newX])); c_base=time.perf_counter()-t0
la_after=lat(lambda q: sc.query(q,10))
print("BASELINE: lat avant=%.1fms, apres=%.1fms  | update=%.3fs"%(la_before,la_after,c_base), flush=True)
del sc; gc.collect()

# HeCBR avant / apres ajout
Bbase=enc(base)
mp=HeCBRMultiProbeIndex(); mp.build(Bbase,num_tables=8,bits_per_table=24,radius=2,hash_seed=7)
Qc=enc(Q)
lh_before=lat(lambda q: mp.query(q,10,256))
t0=time.perf_counter(); Bnew=enc(newX); c_enc=time.perf_counter()-t0
t0=time.perf_counter(); mp2=HeCBRMultiProbeIndex(); mp2.build(np.concatenate([Bbase,Bnew]),num_tables=8,bits_per_table=24,radius=2,hash_seed=7); c_idx=time.perf_counter()-t0
lh_after=lat(lambda q: mp2.query(q,10,256))
print("HECBR-MP: lat avant=%.1fms, apres=%.1fms | update=(enc %.2fs + idx %.2fs)=%.2fs"%(lh_before,lh_after,c_enc,c_idx,c_enc+c_idx), flush=True)
print("DONE truthtell", flush=True)
