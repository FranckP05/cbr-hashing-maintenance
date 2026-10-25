import numpy as np, sys, time, glob
sys.path.insert(0, "/teamspace/studios/this_studio/hecbr_sota_camelyon")
import torch
from data.loader import load_slides
from data.splitter import split_slides, build_split_matrices
from config.seeds import SPLIT_SEED
from config.paths import HECBR_MODEL_PATH
from scripts.train_hecbr_v4 import HashNetV3
from indexes.hecbr_index import HeCBRIndex, HeCBRMultiProbeIndex

device="cuda" if torch.cuda.is_available() else "cpu"
ck=torch.load(HECBR_MODEL_PATH,map_location="cpu"); model=HashNetV3(512,512,64); model.load_state_dict(ck["state_dict"]); model=model.eval().to(device)
sd,sl=load_slides(); sids=sorted(sl.keys()); b0,bv,bt=split_slides(sids,sl,seed=SPLIT_SEED)
Xb,yb,_,_,Xt,yt=build_split_matrices(base_ids=b0+bv,val_ids=[],test_ids=bt,slide_data=sd,slide_labels=sl,normalize_l2=True)
mu=Xb.mean(0); sd_=Xb.std(0)+1e-6
def enc(X,bs=32768):
    o=[]
    for i in range(0,len(X),bs):
        t=torch.tensor((X[i:i+bs]-mu)/sd_,dtype=torch.float32).to(device)
        with torch.no_grad(): o.append((model(t)>=0).cpu().numpy())
    return np.concatenate(o)
print("encodage base 4.4M..."); t0=time.perf_counter(); Bb=enc(Xb); Be=enc(Xt[:300]); print("encode %.1fs"%(time.perf_counter()-t0))
print("n_base",Bb.shape[0], flush=True)

# scan exact complet
ex=HeCBRIndex(); t0=time.perf_counter(); ex.build(Bb); tb=time.perf_counter()-t0
ex.query(Be[0],10) # warm
t0=time.perf_counter(); hm,_=ex.batch_query(Be,10); tx=time.perf_counter()-t0
print("scan exact 4.4M: build=%.1fs lat=%.2fms/query"%(tb,tx*1000/300), flush=True)

for L,m,r in [(8,24,2),(12,24,3),(16,24,3)]:
    mp=HeCBRMultiProbeIndex(); t0=time.perf_counter(); mp.build(Bb,num_tables=L,bits_per_table=m,radius=r,hash_seed=7); tb=time.perf_counter()-t0
    mp.query(Be[0],10,max_buckets_per_table=256 if r==2 else 2048)
    times=[];ncs=[];rech=[]
    for i in range(300):
        t0=time.perf_counter(); idx,_,nc=mp.query(Be[i],10,max_buckets_per_table=256 if r==2 else 2048); times.append(time.perf_counter()-t0); ncs.append(nc)
        rech.append(len(set(hm[i].tolist())&set(idx.tolist()))/10.0)
    print("MP full L=%d m=%d r=%d: build=%.1fs lat=%.2fms/query cand=%.0f recall_vsExactHam=%.3f"%(
        L,m,r,tb,float(np.mean(times))*1000,float(np.mean(ncs)),float(np.mean(rech))), flush=True)
