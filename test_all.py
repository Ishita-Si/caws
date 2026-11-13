import numpy as np, sys
from model import *; from milp import solve_milp; from chain import chain_exact; from gen import *
rng = np.random.default_rng(1); bad = 0
# 1. topgamma vs adversary brute force
for _ in range(200):
    inst = random_instance(rng, n=4, nproc=2, slack=4, Gamma=int(rng.integers(0, 4)))
    S = asap(inst); assert check(inst, S)[0]
    assert abs(worst_case_cost(inst, S) - worst_case_bruteforce(inst, S)) < 1e-6
print("T1 worst-case formula == adversary enumeration: 200/200")
# 2. MILP == brute force (DAG, 2 procs, cap, fractional Gamma uses topgamma)
mx = 0; cnt = 0
for i in range(60):
    G = float(rng.choice([0, 1, 1.5, 2, 3]))
    cap = float(rng.choice([np.inf, 4.0, 5.0]))
    inst = random_instance(rng, n=int(rng.integers(3, 6)), nproc=int(rng.integers(1, 3)), slack=int(rng.integers(0, 4)), Gamma=G, cap=cap)
    if asap(inst) is None: continue
    bf, Sb = brute_force_opt(inst); r = solve_milp(inst)
    assert check(inst, r["S"])[0], check(inst, r["S"])
    assert abs(worst_case_cost(inst, r["S"]) - r["obj"]) < 1e-5 * max(1, bf), (worst_case_cost(inst, r["S"]), r["obj"])
    d = abs(bf - r["obj"]); mx = max(mx, d); cnt += 1
    assert d < 1e-4 * max(1, bf), (bf, r["obj"])
print(f"T2 MILP optimum == brute force on {cnt} random DAG/cap/fractional-Gamma instances, max abs diff {mx:.2e}")
# 3. chain algorithm == MILP == brute force
cnt = 0; mx = 0
for i in range(80):
    inst = random_instance(rng, n=int(rng.integers(2, 6)), nproc=1, slack=int(rng.integers(0, 7)), Gamma=float(rng.choice([0, 1, 2, 2.5, 4])),
                           kind="chain", cap=float(rng.choice([np.inf, 2.0, 3.0])))
    v, S = chain_exact(inst)
    if S is None or not np.isfinite(v): continue
    assert check(inst, S)[0]
    assert abs(worst_case_cost(inst, S) - v) < 1e-6, (worst_case_cost(inst, S), v)
    bf, _ = brute_force_opt(inst); mx = max(mx, abs(bf - v)); cnt += 1
    assert abs(bf - v) < 1e-6 * max(1, bf), (bf, v)
print(f"T3 chain theta-DP == brute force on {cnt} instances, max abs diff {mx:.2e}")
# 4. monotone + concave in Gamma, bounds
for i in range(20):
    inst = random_instance(rng, n=4, nproc=2, slack=4); vals = []
    for G in [0, 1, 2, 3, 4, 5]:
        vals.append(solve_milp(inst, Gamma=G)["obj"])
    d = np.diff(vals); assert np.all(d >= -1e-6) and np.all(np.diff(d) <= 1e-6), vals
print("T4 optimal value nondecreasing and concave in Gamma: 20/20")
