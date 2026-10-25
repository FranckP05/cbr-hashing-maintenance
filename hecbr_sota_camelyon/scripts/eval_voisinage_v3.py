import numpy as np, sys, time
sys.path.insert(0, "/teamspace/studios/this_studio/hecbr_sota_camelyon")
import torch
from data.loader import load_slides
from data.splitter import split_slides, build_split_matrices
from config.seeds import SPLIT_SEED
from config.paths import HECBR_MODEL_PATH
from scripts.train_hecbr_v3 import HashNetV3
from indexes.linear_scan import LinearScan
from indexes.hecbr_index import HeCBRIndex, HeCBRMultiProbeIndex

device="cuda" if torch.cuda.is_available() else "cpu"
ck=torch.load(HECBR_MODEL_PATH, map_location="cpu")
model=HashNetV3(in_dim=512,hidden=512,bit_length=64); model.load_state_dict(ck["state_dict"]); model.eval()
model=model.to(device)

sd,sl=load_slides(); sids=sorted(sl.keys())
b0,bv,bt=split_slides(sids,sl,seed=SPLIT_SEED)
Xb,yb,_,_,Xt,yt=build_split_matrices(base_ids=b0+bv,val_ids=[],test_ids=bt,slide_data=sd,slide_labels=sl,normalize_l2=True)

# encodage sur sous-ensemble base pour vitesse
rng=np.random.default_rng(0)
bsub=rng.choice(np.arange(len(Xb)),300000,replace=False)
Xsub=Xb[bsub]
def enc(X,bs=8192):
    out=[]
    for i in range(0,len(X),bs):
        t=torch.tensor(X[i:i+bs],dtype=torch.float32).to(device)
        with torch.no_grad(): u=model(t); out.append((u>=0).cpu().numpy())
    return np.concatenate(out)
t0=time.perf_counter(); Bb=enc(Xsub); Be=enc(Xt); print("encode %.1fs"%(time.perf_counter()-t0))

# GT L2 top10 sur requetes (exact, base complete pour L2 de reference: comparons sur bsub)
sc=LinearScan(); sc.build(Xsub)
q=enc(Xt[:200])
gt_idx,_=sc.batch_query(Xt[:200],10)  # idx vers base complete -> map vers bsub? non. utilisez Xt vs Xsub
# refaire L2 sur le sous-ensemble directement
gt_idx,_=sc.batch_query(Xt[:200],10)

# GT Hamming exact
ex=HeCBRIndex(); ex.build(Bb)
hm_idx,_=ex.batch_query(q,10)
recall=[]
for i in range(200):
    gt=set(gt_idx[i].tolist())
    got=set(hm_idx[i].tolist())
    recall.append(len(gt&got)/10.0)
print("Recall@10 Hamming (sur bsub 300k) vs L2 = %.3f"%float(np.mean(recall)))

# latence multi-probe
for L,m,r,b in [(8,16,2,256),(16,16,2,256),(16,12,2,256),(8,24,3,2048)]:
    mp=HeCBRMultiProbeIndex(); t0=time.perf_counter(); mp.build(Bb,num_tables=L,bits_per_table=m,radius=r,hash_seed=7); tb=time.perf_counter()-t0
    times=[]; rec=[]; ncs=[]
    for i in range(200):
        t0=time.perf_counter(); idx,_,nc=mp.query(q[i],10,b); times.append(time.perf_counter()-t0); ncs.append(nc)
        rec.append(len(set(gt_idx[i].tolist())&set(idx.tolist()))/10.0)
    print("MP L=%d m=%d r=%d b=%d: lat=%.2fms cand=%.0f RecallVS_L2=%.3f build=%.1fs"%(L,m,r,b,float(np.mean(times))*1000,float(np.mean(ncs)),float(np.mean(rec)),tb))
