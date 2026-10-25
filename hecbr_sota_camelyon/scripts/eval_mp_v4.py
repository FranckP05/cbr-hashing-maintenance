import numpy as np, sys, time
sys.path.insert(0, "/teamspace/studios/this_studio/hecbr_sota_camelyon")
import torch
from data.loader import load_slides
from data.splitter import split_slides, build_split_matrices
from config.seeds import SPLIT_SEED
from config.paths import HECBR_MODEL_PATH
from scripts.train_hecbr_v4 import HashNetV3
from indexes.hecbr_index import HeCBRIndex, HeCBRMultiProbeIndex
from indexes.linear_scan import LinearScan

device="cuda" if torch.cuda.is_available() else "cpu"
ck=torch.load(HECBR_MODEL_PATH,map_location="cpu"); model=HashNetV3(512,512,64); model.load_state_dict(ck["state_dict"]); model=model.eval().to(device)
sd,sl=load_slides(); sids=sorted(sl.keys()); b0,bv,bt=split_slides(sids,sl,seed=SPLIT_SEED)
Xb,yb,_,_,Xt,yt=build_split_matrices(base_ids=b0+bv,val_ids=[],test_ids=bt,slide_data=sd,slide_labels=sl,normalize_l2=True)
rng=np.random.default_rng(0); bsub=rng.choice(np.arange(len(Xb)),300000,replace=False); Xsub=Xb[bsub]
mu=Xsub.mean(0); sd_=Xsub.std(0)+1e-6
def enc(X,bs=32768):
    o=[]
    for i in range(0,len(X),bs):
        t=torch.tensor((X[i:i+bs]-mu)/sd_,dtype=torch.float32).to(device)
        with torch.no_grad(): o.append((model(t)>=0).cpu().numpy())
    return np.concatenate(o)
Bb=enc(Xsub); Be=enc(Xt[:200])

# GT L2 exacte sur bsub
sc=LinearScan(); sc.build(Xsub); gt_idx,_=sc.batch_query(Xt[:200],10)

# scan exact Hamming (baseline HeCBR)
ex=HeCBRIndex(); t0=time.perf_counter(); ex.build(Bb); tb_h=time.perf_counter()-t0
t0=time.perf_counter(); hm,_=ex.batch_query(Be,10); te_h=time.perf_counter()-t0
print("HeCBR scan exact: build=%.2fs q/200=%.1fms (%.2fms/query) cand=all"%(tb_h,te_h*1000,te_h*1000/200))

for L,m,r,b in [(8,24,2,256),(16,24,2,256),(12,24,3,2048),(16,24,3,2048),(12,28,3,2048),(16,20,3,1024)]:
    mp=HeCBRMultiProbeIndex(); t0=time.perf_counter(); mp.build(Bb,num_tables=L,bits_per_table=m,radius=r,hash_seed=7); tb=time.perf_counter()-t0
    mp.query(Be[0],10,b)  # warmup
    times=[]; ncs=[]; rec=[]; rec_h=[]
    for i in range(200):
        t0=time.perf_counter(); idx,_,nc=mp.query(Be[i],10,b); times.append(time.perf_counter()-t0); ncs.append(nc)
        rec.append(len(set(gt_idx[i].tolist()) & set(idx.tolist()))/10.0)
        rec_h.append(len(set(hm[i].tolist()) & set(idx.tolist()))/10.0)
    print("MP L=%d m=%d r=%d b=%d: build=%.1fs lat=%.2fms cand=%.0f recall_vsL2=%.3f recall_vsExactHam=%.3f"%(
        L,m,r,b,tb,float(np.mean(times))*1000,float(np.mean(ncs)),float(np.mean(rec)),float(np.mean(rec_h))))
