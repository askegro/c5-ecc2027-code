import numpy as np
from math import comb
def boot_ci_median(d, n=10000, seed=20261001):
    rng=np.random.default_rng(seed); d=np.asarray(d,float)
    m=np.median(d[rng.integers(0,len(d),(n,len(d)))],axis=1)
    return [float(np.percentile(m,2.5)),float(np.percentile(m,97.5))]
def sign_p(d):
    d=np.asarray(d,float); pos=int((d>0).sum()); neg=int((d<0).sum()); n=pos+neg
    if n==0: return 1.0
    k=min(pos,neg); return float(min(1.0,2*sum(comb(n,i) for i in range(k+1))/2**n))
def holm(ps):
    ps=np.asarray(ps,float); o=np.argsort(ps); adj=np.empty(len(ps)); run=0.0
    for r,i in enumerate(o):
        run=max(run,(len(ps)-r)*ps[i]); adj[i]=min(1.0,run)
    return adj.tolist()
