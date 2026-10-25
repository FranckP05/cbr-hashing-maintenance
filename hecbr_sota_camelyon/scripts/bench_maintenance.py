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
from indexes.cross_polytope import CrossPolytopeLSH

device="cuda" if torch.cuda.is_available() else "cpu"
ck=torch.load(HECBR_MODEL_PATH,map_location="cpu"); model=HashNetV3(512,512,64); model.load_state_dict(ck["state_dict"]); model=model.eval().to(device)
sd,sl=load_slides(); sids=sorted(sl.keys()); b0,bv,bt=split_slides(sids,sl,seed=SPLIT_SEED)
Xb,yb,_,_,Xt,yt=build_split_matrices(base_ids=b0+bv,val_ids=[],test_ids=bt,slide_data=sd,slide_labels=sl,normalize_l2=True)
mu=Xb.mean(0); sd_=Xb.std(0)+1e-6
def enc(X,bs=65536):
    o=[]
    for i in range(0,len(X),bs):
        t=torch.tensor((X[i:i+bs]-mu)/sd_,dtype=torch.float32).to(device)
        with torch.no_grad(): o.append((model(t)>=0).cpu().numpy())
        del t
    return np.concatenate(o)

print("===== A. SCALING DE LA LATENCE QUERY QUAND N CROIT =====", flush=True)
rng=np.random.default_rng(0)
print("one-time encode full base...", flush=True); t0=time.perf_counter(); B_all=enc(Xb); print("  encode full = %.1fs"%(time.perf_counter()-t0), flush=True)
q_idx=rng.choice(len(Xb),100,replace=False); Qv=Xb[q_idx]; qcode=enc(Qv)

for N in [500000, 1000000, 2000000, 4416441]:
    idx=rng.choice(len(Xb),N,replace=False)
    Xn=Xb[idx]; Bn=B_all[idx]
    # baseline exact
    sc=LinearScan(); t0=time.perf_counter(); sc.build(Xn); bl_build=time.perf_counter()-t0
    sc.query(Qv[0],10); tt=[sc.query(Qv[i],10) for i in range(0,30,5)]; bl=float(np.median([time.perf_counter()-t0 for _ in [] ]))
    tlist=[]
    for i in range(0,30,5):
        t0=time.perf_counter(); sc.query(Qv[i],10); tlist.append(time.perf_counter()-t0)
    del sc; gc.collect(); bl=float(np.median(tlist))*1000
    # LSH aleatoire
    cp=CrossPolytopeLSH(num_hashes=2,num_tables=8,num_rotations=2,num_probes=8,hash_seed=11)
    t0=time.perf_counter(); cp.build(Xn); lsh_build=time.perf_counter()-t0
    cp.query(Qv[0],10); tlist=[]
    for i in range(0,30,5):
        t0=time.perf_counter(); cp.query(Qv[i],10); tlist.append(time.perf_counter()-t0)
    del cp; gc.collect(); lsh=float(np.median(tlist))*1000
    # HeCBR multi-probe (data-dependent)
    mp=HeCBRMultiProbeIndex(); t0=time.perf_counter(); mp.build(Bn,num_tables=8,bits_per_table=24,radius=2,hash_seed=7); hb=time.perf_counter()-t0
    mp.query(qcode[0],10,256); tlist=[]
    for i in range(0,30,5):
        t0=time.perf_counter(); mp.query(qcode[i],10,256); tlist.append(time.perf_counter()-t0)
    del mp; gc.collect(); hmp=float(np.median(tlist))*1000
    print("N=%7d | baseline build=%.3fs lat(med)=%.1fms | LSH build=%.1fs lat=%.1fms | HeCBR-MP build=%.2fs lat=%.1fms"%(N,bl_build,bl,lsh_build,lsh,hb,hmp), flush=True)

print("===== B. COUT DE MAINTENANCE : INSERTION DE 10k NOUVEAUX POINTS =====", flush=True)
newX=Xb[rng.choice(len(Xb),10000,replace=False)]
t0=time.perf_counter(); newcode=enc(newX); c_enc=time.perf_counter()-t0
print("HeCBR: encoder 10k nouveaux = %.2fs (<= cout data-dependent)"%c_enc, flush=True)

# baseline : ajout de points a la base (concat) + rebuild scan
base=Xb[:1000000]
t0=time.perf_counter(); base2=np.concatenate([base,newX]); c_base=time.perf_counter()-t0
print("Baseline: ajouter 10k pts (concat) = %.3fs (mais query reste O(N*d))"%c_base, flush=True)

# LSH : ajout = rebucket (re-build) car detecte; mesure rebuild
cp=CrossPolytopeLSH(num_hashes=2,num_tables=8,num_rotations=2,num_probes=8,hash_seed=11)
cp.build(base)
t0=time.perf_counter(); cp.build(np.concatenate([base,newX])); c_lsh=time.perf_counter()-t0
print("LSH: re-hachage/rebuild apres +10k = %.2fs"%c_lsh, flush=True)

# HeCBR : re-encode (déjà via newcode) + rebuild de l'index mp sur base+codes
bc=B_all[:1000000]; t0=time.perf_counter(); mp=HeCBRMultiProbeIndex(); mp.build(np.concatenate([bc,newcode]),num_tables=8,bits_per_table=24,radius=2,hash_seed=7); c_mp=time.perf_counter()-t0
print("HeCBR: rebuild index apres +10k = %.2fs (encoder %.2fs + re-index %.2fs)"%(c_enc+c_mp,c_enc,c_mp), flush=True)
print("DONE", flush=True)
