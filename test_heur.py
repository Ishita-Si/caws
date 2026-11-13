import numpy as np, time
from model import *; from milp import solve_milp; from heur import *; from gen import *
rng = np.random.default_rng(7); gaps = []; nc = 0
for i in range(80):
    inst = random_instance(rng, n=int(rng.integers(4, 9)), nproc=int(rng.integers(1, 3)), slack=int(rng.integers(2, 8)),
                           Gamma=float(rng.choice([0, 1, 2, 3])), cap=float(rng.choice([np.inf, 5.0, 6.0])))
    if asap(inst) is None: nc += 1; continue
    S = robslack(inst); assert check(inst, S)[0], check(inst, S)
    r = solve_milp(inst); h = worst_case_cost(inst, S); a = worst_case_cost(inst, asap(inst))
    assert h <= a + 1e-9
    gaps.append((h - r["obj"]) / r["obj"])
gaps = np.array(gaps); print("n", len(gaps), "mean gap %.3f%% max %.3f%% optimal in %d" % (100*gaps.mean(), 100*gaps.max(), (gaps < 1e-9).sum()))
