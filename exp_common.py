import numpy as np, json, time, sys
from model import *; from heur import *; from data import *; from gen import random_dag
def workflow(seed, n=30, nproc=4, dmax=3):
    rng = np.random.default_rng(seed)
    dur = list(rng.integers(1, dmax + 1, n)); pw = [float(x) for x in rng.choice([1, 2, 3, 4], n)]
    proc = list(rng.integers(0, nproc, n)); edges = random_dag(rng, n, layers=max(3, int(np.sqrt(n))), p_edge=0.25)
    return dur, pw, proc, edges
def make_inst(wf, slack, c, e, Gamma=0.0, cap=np.inf, Dmax=H):
    dur, pw, proc, edges = wf
    big = Instance(dur, pw, proc, edges, 500, np.zeros(500), np.zeros(500))
    ms = makespan(big, asap(big)); D = min(Dmax, ms + slack)
    return Instance(dur, pw, proc, edges, D, np.asarray(c, float), np.asarray(e, float), Gamma, cap), ms
def data():
    A, F = load_csv("uk_carbon.csv")
    e = calibrate(A[:60], F[:60]); return A[60:100], F[60:100], e

