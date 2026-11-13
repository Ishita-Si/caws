"""Exact polynomial algorithm: chain (or any set of tasks with a fixed order) on ONE processor, Gamma-budget uncertainty.
Theta-threshold DP. Candidates theta in {0} U {e_t * p_v}."""
import numpy as np
from model import *

def chain_exact(inst, order=None, return_theta=False):
    n, D = inst.n, inst.D
    assert len(set(inst.proc)) == 1
    order = order or inst.topo()
    c, e, G = inst.c[:D], inst.e[:D], inst.Gamma
    thetas = sorted(set([0.0] + [float(e[t] * p) for t in range(D) for p in set(inst.pw)]))
    best = (np.inf, None, None)
    for th in thetas:
        # per-task per-start cost for this theta (separable because one task per slot)
        prev = np.zeros(D + 1)   # prev[s]: min cost of tasks so far with finishing time <= s
        back = []
        INF = np.inf
        dp_prev = np.zeros(D + 1)  # best cost with all previous tasks finished by time s (non-decreasing in s)
        choice = []
        for v in order:
            d, p = inst.dur[v], inst.pw[v]
            if p > inst.cap + 1e-9: dp_prev = np.full(D + 1, INF); break
            slot = c * p + np.maximum(e * p - th, 0)
            cs = np.concatenate([[0], np.cumsum(slot)])
            f = np.full(D + 1, INF); ch = np.full(D + 1, -1)
            for s in range(0, D - d + 1):
                val = dp_prev[s] + cs[s + d] - cs[s]
                if val < f[s + d]: f[s + d] = val; ch[s + d] = s
            # allow idle: finishing by time T
            for T in range(1, D + 1):
                if f[T - 1] < f[T]: f[T] = f[T - 1]; ch[T] = -2
            choice.append(ch); dp_prev = f
        tot = dp_prev[D] + G * th
        if tot < best[0] - 1e-12:
            # recover
            S = [None] * n; T = D
            for i in range(n - 1, -1, -1):
                v = order[i]; ch = choice[i]
                while ch[T] == -2: T -= 1
                S[v] = int(ch[T]); T = S[v]
            best = (tot, S, th)
    val = inst.h * best[0]
    return (val, best[1], best[2]) if return_theta else (val, best[1])
