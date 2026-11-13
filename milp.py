"""Exact robust MILP (Gamma-budget dual linearisation) with scipy/HiGHS."""
import numpy as np, time
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix
from model import *

def solve_milp(inst, Gamma=None, time_limit=60, lp_relax=False, fixed=None, release=0, mip_gap=1e-6):
    G = inst.Gamma if Gamma is None else Gamma
    n, D = inst.n, inst.D; fixed = fixed or {}
    # variable index: z[v,s] for s in 0..D-dur ; then theta ; then q[t]
    idx = {}; k = 0
    for v in range(n):
        for s in range(D - inst.dur[v] + 1):
            idx[(v, s)] = k; k += 1
    nz = k; th = k; k += 1; q0 = k; k += D; N = k
    cost = np.zeros(N); lb = np.zeros(N); ub = np.ones(N)
    c = inst.c[:D]; e = inst.e[:D]
    # P(t) coefficients
    Prow = [dict() for _ in range(D)]
    for (v, s), j in idx.items():
        for t in range(s, s + inst.dur[v]): Prow[t][j] = inst.pw[v]
        for t in range(s, s + inst.dur[v]): cost[j] += c[t] * inst.pw[v]
    cost[th] = G; ub[th] = np.inf
    cost[q0:q0 + D] = 1.0; ub[q0:q0 + D] = np.inf
    rows = []; lo = []; hi = []
    A = lil_matrix((n + len(inst.edges) + 2 * D + D + 0, N)); r = 0
    def add(lo_, hi_):
        nonlocal r; lo.append(lo_); hi.append(hi_); r += 1; return r - 1
    # each task once
    for v in range(n):
        for s in range(D - inst.dur[v] + 1): A[r, idx[(v, s)]] = 1
        add(1, 1)
    for v, s0 in fixed.items():
        for s in range(D - inst.dur[v] + 1):
            if s != s0: ub[idx[(v, s)]] = 0
    for v in range(n):
        if v in fixed: continue
        for s in range(min(release, D - inst.dur[v] + 1)): ub[idx[(v, s)]] = 0
    # precedence: sum s z_v - sum s z_u >= dur_u
    for u, v in inst.edges:
        for s in range(D - inst.dur[v] + 1): A[r, idx[(v, s)]] += s
        for s in range(D - inst.dur[u] + 1): A[r, idx[(u, s)]] -= s
        add(inst.dur[u], np.inf)
    # processor: at most one task per processor per slot
    byp = {}
    for v in range(n): byp.setdefault(inst.proc[v], []).append(v)
    Ap = []
    for p, vs in byp.items():
        if len(vs) < 2: continue
        for t in range(D):
            cols = [idx[(v, s)] for v in vs for s in range(max(0, t - inst.dur[v] + 1), min(t, D - inst.dur[v]) + 1)]
            if len(cols) > 1: Ap.append((cols, 1.0))
    # cap and robust rows
    Acap = []
    if np.isfinite(inst.cap):
        for t in range(D):
            if Prow[t]: Acap.append((Prow[t], inst.cap))
    nrow = r + len(Ap) + len(Acap) + D
    A2 = lil_matrix((nrow, N)); A2[:r, :] = A[:r, :]
    lo2 = list(lo); hi2 = list(hi); rr = r
    for cols, b in Ap:
        for j in cols: A2[rr, j] = 1
        lo2.append(-np.inf); hi2.append(b); rr += 1
    for d, b in Acap:
        for j, w in d.items(): A2[rr, j] = w
        lo2.append(-np.inf); hi2.append(b); rr += 1
    for t in range(D):  # q_t + theta - e_t P_t >= 0
        for j, w in Prow[t].items(): A2[rr, j] = -e[t] * w
        A2[rr, th] = 1; A2[rr, q0 + t] = 1
        lo2.append(0); hi2.append(np.inf); rr += 1
    integrality = np.zeros(N); 
    if not lp_relax: integrality[:nz] = 1
    t0 = time.time()
    res = milp(cost * inst.h, constraints=LinearConstraint(A2.tocsr(), lo2, hi2), integrality=integrality,
               bounds=Bounds(lb, ub), options={"time_limit": time_limit, "mip_rel_gap": mip_gap, "presolve": True})
    el = time.time() - t0
    if res.x is None: return dict(status=res.status, S=None, obj=None, time=el, bound=None)
    S = None
    if not lp_relax:
        S = [int(round(sum(s * res.x[idx[(v, s)]] for s in range(D - inst.dur[v] + 1)))) for v in range(n)]
    return dict(status=res.status, S=S, obj=res.fun, time=el, bound=getattr(res, "mip_dual_bound", None), optimal=(res.status == 0))
