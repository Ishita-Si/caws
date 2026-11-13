"""RobSlack: threshold-cost construction + true-objective local search, with ASAP fallback (never worse than ASAP)."""
import numpy as np
from model import *

def tails(inst):
    sc = inst.succs(); t = [0] * inst.n
    for v in reversed(inst.topo()): t[v] = inst.dur[v] + max([t[w] for w in sc[v]], default=0)
    return t

def _wc(inst, P, G):
    return float(inst.c[:inst.D] @ P) + topgamma(inst.e[:inst.D] * P, G)

def construct(inst, theta, G, release=0, fixed=None):
    """Place tasks in least-slack-first topological order at the start minimising the theta-threshold task cost."""
    D, n = inst.D, inst.n; fixed = fixed or {}
    pr = inst.preds(); tl = tails(inst); topo = inst.topo(); rank = upward_rank(inst)
    c, e = inst.c[:D], inst.e[:D]
    S = [None] * n; P = np.zeros(D); occ = {}
    def put(v, s):
        S[v] = s; P[s:s + inst.dur[v]] += inst.pw[v]
        for t in range(s, s + inst.dur[v]): occ[(inst.proc[v], t)] = v
    for v, s in fixed.items(): put(v, s)
    done = set(fixed)
    ready_order = []
    # dynamic: repeatedly pick the ready task with smallest latest-start
    remaining = [v for v in range(n) if v not in done]
    while remaining:
        ready = [v for v in remaining if all(S[u] is not None for u in pr[v])]
        v = min(ready, key=lambda x: (D - tl[x], -rank[x]))
        est = max([release] + [S[u] + inst.dur[u] for u in pr[v]]); ls = D - tl[v]
        d, p = inst.dur[v], inst.pw[v]
        slot = c * p + (np.maximum(e * p - theta, 0) if np.isfinite(theta) else 0)
        cs = np.concatenate([[0], np.cumsum(slot)])
        cand = []
        for s in range(est, D - d + 1):
            if any((inst.proc[v], t) in occ for t in range(s, s + d)): continue
            if np.any(P[s:s + d] + p > inst.cap + 1e-9): continue
            cand.append((cs[s + d] - cs[s], s))
        cand.sort()
        best = None
        placed = {u: S[u] for u in range(n) if S[u] is not None}
        for val, s in cand:   # cheapest first; accept the first start for which the rest can still be completed (ASAP check)
            trial = dict(placed); trial[v] = s
            if list_schedule(inst, topo, release, trial) is not None: best = (val, s); break
        if best is None: return None
        put(v, best[1]); remaining.remove(v)
    return S

def local_search(inst, S, G, release=0, fixed=None, max_pass=20):
    D, n = inst.D, inst.n; fixed = fixed or {}
    pr, sc = inst.preds(), inst.succs()
    S = list(S); P = power_profile(inst, S); cur = _wc(inst, P, G)
    occ = {}
    for v in range(n):
        for t in range(S[v], S[v] + inst.dur[v]): occ[(inst.proc[v], t)] = v
    for _ in range(max_pass):
        improved = False
        for v in range(n):
            if v in fixed: continue
            d, p = inst.dur[v], inst.pw[v]
            lo = max([release] + [S[u] + inst.dur[u] for u in pr[v]])
            hi = min([D] + [S[w] for w in sc[v]]) - d
            Pb = P.copy(); Pb[S[v]:S[v] + d] -= p
            best = (cur, S[v])
            for s in range(lo, hi + 1):
                if s == S[v]: continue
                if any(occ.get((inst.proc[v], t), v) != v for t in range(s, s + d)): continue
                if np.any(Pb[s:s + d] + p > inst.cap + 1e-9): continue
                Pn = Pb.copy(); Pn[s:s + d] += p
                val = _wc(inst, Pn, G)
                if val < best[0] - 1e-9: best = (val, s)
            if best[1] != S[v]:
                for t in range(S[v], S[v] + d): occ.pop((inst.proc[v], t), None)
                S[v] = best[1]
                for t in range(S[v], S[v] + d): occ[(inst.proc[v], t)] = v
                P = Pb; P[S[v]:S[v] + d] += p; cur = best[0]; improved = True
        if not improved: break
    return S, cur

def robslack(inst, Gamma=None, release=0, fixed=None, quantiles=(0.0, 0.25, 0.5, 0.75, 0.9), use_inf=True, ils_rounds=30):
    G = inst.Gamma if Gamma is None else Gamma
    D = inst.D; e = inst.e[:D]
    vals = np.array(sorted({float(e[t] * p) for t in range(D) for p in set(inst.pw) if e[t] * p > 0}))
    thetas = [0.0] if len(vals) == 0 else [float(np.quantile(vals, q)) for q in quantiles]
    if use_inf: thetas.append(np.inf)
    cands = []
    base = asap(inst, release, fixed)
    if base is not None: cands.append(base)
    for th in thetas:
        S = construct(inst, th, G, release, fixed)
        if S is not None: cands.append(S)
    if not cands: return None
    best = (np.inf, None)
    for S in cands:
        S2, val = local_search(inst, S, G, release, fixed)
        if val < best[0]: best = (val, S2)
    # iterated local search: random kicks (move 2 tasks to random feasible starts), re-optimise, keep if better
    rng = np.random.default_rng(12345); fx = fixed or {}
    free = [v for v in range(inst.n) if v not in fx]
    pr, sc = inst.preds(), inst.succs()
    for it in range(ils_rounds):
        if not free: break
        S = list(best[1]); occ = {}
        for v in range(inst.n):
            for t in range(S[v], S[v] + inst.dur[v]): occ[(inst.proc[v], t)] = v
        P = power_profile(inst, S)
        for v in rng.choice(free, size=min(2, len(free)), replace=False):
            d, p = inst.dur[v], inst.pw[v]
            lo = max([release] + [S[u] + inst.dur[u] for u in pr[v]]); hi = min([D] + [S[w] for w in sc[v]]) - d
            Pb = P.copy(); Pb[S[v]:S[v] + d] -= p
            opts = [s2 for s2 in range(lo, hi + 1) if all(occ.get((inst.proc[v], t), v) == v for t in range(s2, s2 + d))
                    and np.all(Pb[s2:s2 + d] + p <= inst.cap + 1e-9)]
            if not opts: continue
            s2 = int(rng.choice(opts))
            for t in range(S[v], S[v] + d): occ.pop((inst.proc[v], t), None)
            S[v] = s2; P = Pb; P[s2:s2 + d] += p
            for t in range(s2, s2 + d): occ[(inst.proc[v], t)] = v
        S2, val = local_search(inst, S, G, release, fixed)
        if val < best[0] - 1e-9: best = (val, S2)
    if base is not None:
        assert best[0] <= _wc(inst, power_profile(inst, base), G) + 1e-9  # no regression against ASAP
    return best[1]

def greedy_cheapest_window(inst, release=0, fixed=None):
    """Carbon-aware but forecast-trusting: theta=inf construction, no local search, Gamma=0."""
    S = construct(inst, np.inf, 0.0, release, fixed)
    return S if S is not None else asap(inst, release, fixed)

def nominal_heuristic(inst, release=0, fixed=None):
    """RobSlack with Gamma=0: forecast-optimal plan (large instances)."""
    return robslack(inst, Gamma=0.0, release=release, fixed=fixed)
