import numpy as np, sys
sys.path.insert(0, "/teamspace/studios/this_studio/hecbr_sota_camelyon")
import torch
from data.loader import load_slides
from data.splitter import split_slides, build_split_matrices
from config.seeds import SPLIT_SEED
from config.paths import HECBR_MODEL_PATH
from scripts.train_hecbr_v4 import HashNetV3
device="cuda" if torch.cuda.is_available() else "cpu"
ck=torch.load(HECBR_MODEL_PATH,map_location="cpu")
model=HashNetV3(512,512,64); model.load_state_dict(ck["state_dict"]); model.eval(); model=model.to(device)
sd,sl=load_slides(); sids=sorted(sl.keys())
b0,bv,bt=split_slides(sids,sl,seed=SPLIT_SEED)
Xb,yb,_,_,Xt,yt=build_split_matrices(base_ids=b0+bv,val_ids=[],test_ids=bt,slide_data=sd,slide_labels=sl,normalize_l2=True)
rng=np.random.default_rng(0)
idx=rng.choice(np.arange(len(Xb)),20000,replace=False)
Xm=Xb[idx]
mu=Xm.mean(0); sd_=Xm.std(0)+1e-6
def norm(X): return (X-mu)/sd_
def enc(X,bs=8192):
    out=[]
    for i in range(0,len(X),bs):
        t=torch.tensor(norm(X[i:i+bs]),dtype=torch.float32).to(device)
        with torch.no_grad(): u=model(t); out.append((u>=0).cpu().numpy())
    return np.concatenate(out)
B=enc(Xm)
C=Xm@Xm.T
r=np.random.default_rng(1); ii=r.integers(0,20000,(5000,)); jj=r.integers(0,20000,(5000,))
coss=C[ii,jj]
hd=(B[ii].astype(np.int8)!=B[jj].astype(np.int8)).sum(1).astype(float)
corr=np.corrcoef(coss,hd)[0,1]
print("correlation(cos,hamming) train = %.4f"%corr)
from indexes.hecbr_index import HeCBRIndex
ex=HeCBRIndex(); ex.build(B)
top=300
hm,_=ex.batch_query(B[:top],100)
gtL=(-C[:top]).argpartition(100,axis=1)[:,:100]
rec={}
for kk in [10,50,100]:
    rk=[len(set(gtL[i,:kk].tolist()) & set(hm[i,:kk].tolist()))/kk for i in range(top)]
    rec[kk]=float(np.mean(rk))
print("train recall@k hamming vs l2:", {k:round(v,3) for k,v in rec.items()})
bins=np.linspace(coss.min(),coss.max(),6)
print("cos-bin -> hamming moyen:")
for b in range(5):
    m=(coss>=bins[b])&(coss<bins[b+1])
    if m.sum(): print("  cos[%.2f,%.2f]: hd=%.1f"%(bins[b],bins[b+1],hd[m].mean()))
