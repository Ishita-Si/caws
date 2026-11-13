import numpy as np, json
from exp_common import *; from chain import chain_exact
A, F, e = data(); out = {}
rng = np.random.default_rng(3)
for k in [1, 2, 4]:
    for slack in [8, 16, 24]:
        rec = {G: [] for G in [0, 1, 2, 3, 4, 6]}; asapv = []
        for rep in range(10):
            dur = list(rng.integers(2, 4, k)); pw = [1.0] * k; edges = [(i, i + 1) for i in range(k - 1)]
            ms = sum(dur)
            for d in range(40):
                base = Instance(dur, pw, [0] * k, edges, ms + slack, F[d], e, 0.0)
                a = actual_cost(base, asap(base), A[d]); asapv.append(a)
                for G in rec:
                    inst = Instance(dur, pw, [0] * k, edges, ms + slack, F[d], e, float(G)); _, S = chain_exact(inst)
                    rec[G].append(actual_cost(inst, S, A[d]))
        a = np.array(asapv)
        out[f"k{k}_slack{slack}"] = {str(G): dict(saving=100 * (1 - np.sum(v) / a.sum()), worse=100 * float(np.mean(np.array(v) > a + 1e-9)),
                                 p90=float(np.quantile(np.array(v) / a, 0.9)), p99=float(np.quantile(np.array(v) / a, 0.99)), mx=float((np.array(v) / a).max())) for G, v in rec.items()}
        print(k, slack, {G: (round(x["saving"], 1), round(x["worse"], 1), round(x["p99"], 3)) for G, x in out[f"k{k}_slack{slack}"].items()}, flush=True)
json.dump(out, open("e7.json", "w"), indent=1)
