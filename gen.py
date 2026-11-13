import numpy as np
from model import *
def random_dag(rng, n, layers=None, p_edge=0.3):
    layers = layers or max(2, int(np.sqrt(n)))
    lay = np.sort(rng.integers(0, layers, n)); edges = []
    for v in range(n):
        for u in range(v):
            if lay[u] < lay[v] and rng.random() < p_edge: edges.append((u, v))
    return edges
def random_instance(rng, n=6, nproc=2, slack=6, Gamma=1, cap=None, T_extra=0, kind="dag", dmax=3, p_edge=0.35, c=None, e=None):
    dur = list(rng.integers(1, dmax + 1, n)); pw = list(rng.choice([1.0, 2.0, 3.0], n))
    proc = list(rng.integers(0, nproc, n))
    edges = [(i, i + 1) for i in range(n - 1)] if kind == "chain" else random_dag(rng, n, p_edge=p_edge)
    inst0 = Instance(dur, pw, proc, edges, 10**6, np.zeros(1), np.zeros(1))
    S0 = list_schedule(Instance(dur, pw, proc, edges, 500, np.zeros(500), np.zeros(500)), inst0.topo())
    ms = makespan(inst0, S0); D = ms + slack
    cc = rng.uniform(50, 400, D) if c is None else c[:D]
    ee = rng.uniform(0, 1, D) * cc * 0.8 if e is None else e[:D]
    inst = Instance(dur, pw, proc, edges, D, cc, ee, Gamma, np.inf if cap is None else cap)
    return inst
