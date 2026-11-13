import numpy as np, pandas as pd, sys
from exp_common import *; from chain import chain_exact
df = pd.read_csv('uk_carbon.csv', parse_dates=['timestamp'])
a = df.actual.values; f = df.forecast.values
nd = (len(a) - H) // 24
A0 = np.stack([a[d*24:d*24+H] for d in range(nd)]); F0 = np.stack([f[d*24:d*24+H] for d in range(nd)])
ok = [d for d in range(nd) if (df.timestamp.iloc[d*24+H-1]-df.timestamp.iloc[d*24]).total_seconds()==(H-1)*3600]
A0, F0 = A0[ok], F0[ok]; ntr = 60
rows=[]
for kk in [1, 3, 6]:
    A = F0 + kk*(A0-F0); A = np.clip(A, 5, None)          # amplified real errors (sensitivity analysis)
    e = np.quantile(np.abs(A[:ntr]-F0[:ntr]), 0.9, axis=0)
    At, Ft = A[ntr:ntr+40], F0[ntr:ntr+40]
    # single-task chain (exact)
    rng = np.random.default_rng(3); res = {G: [] for G in [0,2,4]}; asapv=[]
    for rep in range(10):
        dur=[int(rng.integers(2,4))]; 
        for d in range(len(At)):
            base = Instance(dur,[1.0],[0],[],dur[0]+16,Ft[d],e,0.0); asapv.append(actual_cost(base,asap(base),At[d]))
            for G in res:
                inst=Instance(dur,[1.0],[0],[],dur[0]+16,Ft[d],e,float(G)); _,S=chain_exact(inst); res[G].append(actual_cost(inst,S,At[d]))
    asapv=np.array(asapv)
    for G in res:
        r=np.array(res[G])/asapv; print(f"1-task k={kk} G={G}: worse {100*np.mean(r>1+1e-9):.1f}%  p99 {np.quantile(r,.99):.3f} max {r.max():.3f} saving {100*(1-np.sum(res[G])/asapv.sum()):.1f}%", flush=True)
    # large workflow
    wf = workflow(1, n=30); tot={G:0 for G in [0,4,8]}; worse={G:0 for G in tot}; tasap=0; mx={G:0 for G in tot}
    for d in range(0,40,2):
        base,_=make_inst(wf,12,Ft[d],e,0.0); ac=actual_cost(base,asap(base),At[d]); tasap+=ac
        for G in tot:
            inst,_=make_inst(wf,12,Ft[d],e,float(G)); c=actual_cost(inst,robslack(inst,ils_rounds=6),At[d]); tot[G]+=c; worse[G]+=c>ac; mx[G]=max(mx[G],c/ac)
    for G in tot: print(f"DAG30 k={kk} G={G}: saving {100*(1-tot[G]/tasap):.2f}%  worse {100*worse[G]/20:.0f}%  max {mx[G]:.3f}", flush=True)
