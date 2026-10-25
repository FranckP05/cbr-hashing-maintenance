import numpy as np, sys
sys.path.insert(0, "/teamspace/studios/this_studio/hecbr_sota_camelyon")
from data.loader import load_slides
from data.splitter import split_slides, build_split_matrices
from config.seeds import SPLIT_SEED
import torch, torch.nn as nn
from sklearn.metrics import f1_score
sd,sl=load_slides(); sids=sorted(sl.keys())
b0,bv,bt=split_slides(sids,sl,seed=SPLIT_SEED)
Xb,yb,_,_,Xt,yt=build_split_matrices(base_ids=b0+bv,val_ids=[],test_ids=bt,slide_data=sd,slide_labels=sl,normalize_l2=True)
print("balances train",yb.sum(),len(yb)-yb.sum(),"test",yt.sum(),len(yt)-yt.sum())
rng=np.random.default_rng(0)
tr=np.concatenate([rng.choice(np.flatnonzero(yb==1),80000,replace=False), rng.choice(np.flatnonzero(yb==0),80000,replace=False)])
rng.shuffle(tr)
te=np.concatenate([rng.choice(np.flatnonzero(yt==1),3000,replace=False), rng.choice(np.flatnonzero(yt==0),3000,replace=False)])
rng.shuffle(te)
Xtr=torch.tensor(Xb[tr],dtype=torch.float32); ytr=torch.tensor(yb[tr],dtype=torch.long)
Xte=torch.tensor(Xt[te],dtype=torch.float32); yte=torch.tensor(yt[te],dtype=torch.long)
mu=Xtr.mean(0); sd_=Xtr.std(0)+1e-6
def norm(X): return (X-mu)/sd_
Xtr=norm(Xtr); Xte=norm(Xte)
torch.manual_seed(0)
net=nn.Sequential(nn.Linear(512,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Linear(128,2))
opt=torch.optim.Adam(net.parameters(),lr=1e-3); lossf=nn.CrossEntropyLoss()
BSZ=1024
for ep in range(15):
    perm=torch.randperm(len(Xtr)); net.train(); tot=0
    for i in range(0,len(Xtr),BSZ):
        b=perm[i:i+BSZ]; xb_=Xtr[b]; yb_=ytr[b]
        opt.zero_grad(); out=net(xb_); loss=lossf(out,yb_); loss.backward(); opt.step(); tot+=loss.item()*len(b)
    net.eval()
    with torch.no_grad(): pr=net(Xte).argmax(1).numpy()
    print("ep%d train=%.4f test_F1=%.4f acc=%.4f"%(ep,tot/len(Xtr),f1_score(yte.numpy(),pr,average='macro'),float((pr==yte.numpy()).mean())))
