import numpy as np, json, time, sys
from exp_common import *; from milp import solve_milp; from chain import chain_exact
R = {}; t00 = time.time()
def log(*a): print("[%5.0fs]" % (time.time() - t00), *a, flush=True)
A, F, e = data()
NDAYS = 40; ILS = 30

# ---------- E1: small-instance optimality gaps vs exact MILP ----------
log("E1 small-instance gaps")
rng = np.random.default_rng(5); rows = []
for i in range(40):
    n = int(rng.choice([8, 10, 12])); wf = workflow(100 + i, n=n, nproc=3, dmax=2)
    d = i % NDAYS; slack = int(rng.choice([4, 6, 8])); G = float(rng.choice([2, 3, 4]))
    inst, ms = make_inst(wf, slack, F[d], e, G, Dmax=H)
    r = solve_milp(inst, time_limit=60)
    if r["S"] is None: continue
    S = robslack(inst, ils_rounds=ILS)
    row = dict(n=n, D=inst.D, Gamma=G, opt=r["obj"], optimal=bool(r["optimal"]), t_milp=r["time"],
               rob=worst_case_cost(inst, S), asap=worst_case_cost(inst, asap(inst)), heft=worst_case_cost(inst, heft_like(inst)),
               greedy=worst_case_cost(inst, greedy_cheapest_window(inst)),
               nomopt=worst_case_cost(inst, solve_milp(inst, Gamma=0, time_limit=30)["S"]))
    rows.append(row)
R["E1"] = rows; log("E1 done", len(rows))

# ---------- E2: held-out days; Gamma x slack sweep ----------
log("E2 Gamma x slack"); GAM = [0, 1, 2, 4, 8]; SLK = [2, 6, 12, 24]; WFS = [workflow(s, n=30) for s in (1, 2, 3)]
E2 = []
for wi, wf in enumerate(WFS):
    for slack in SLK:
        for d in range(NDAYS):
            base, ms = make_inst(wf, slack, F[d], e, 0.0)
            plans = {"ASAP": asap(base), "HEFT": heft_like(base), "Greedy": greedy_cheapest_window(base)}
            for G in GAM:
                inst, _ = make_inst(wf, slack, F[d], e, float(G))
                plans[f"Rob{G}"] = robslack(inst, ils_rounds=ILS)
            rec = dict(wf=wi, slack=slack, day=d, D=base.D)
            for k, S in plans.items():
                assert check(base, S)[0]
                rec[k] = dict(actual=actual_cost(base, S, A[d]), nominal=nominal_cost(base, S),
                              wc={str(G): worst_case_cost(base, S, G) for G in GAM})
            E2.append(rec)
    log("E2 wf", wi)
R["E2"] = E2

# ---------- E3: power cap ----------
log("E3 power cap"); E3 = []
for wi, wf in enumerate(WFS[:2]):
    for rho in [np.inf, 0.9, 0.75, 0.6, 0.5]:
        for d in range(NDAYS):
            inst0, ms = make_inst(wf, 12, F[d], e, 4.0)
            S0 = asap(inst0); peak = power_profile(inst0, S0).max()
            cap = np.inf if not np.isfinite(rho) else max(rho * peak, max(inst0.pw))
            inst, _ = make_inst(wf, 12, F[d], e, 4.0, cap=cap)
            Sa = asap(inst)
            if Sa is None: continue
            Sr = robslack(inst, ils_rounds=ILS); Sn = robslack(inst, Gamma=0.0, ils_rounds=ILS)
            assert check(inst, Sr)[0] and check(inst, Sn)[0] and check(inst, Sa)[0]
            E3.append(dict(wf=wi, rho=None if not np.isfinite(rho) else rho, day=d, cap=None if not np.isfinite(cap) else cap,
                           ASAP=actual_cost(inst, Sa, A[d]), Rob4=actual_cost(inst, Sr, A[d]), Nom=actual_cost(inst, Sn, A[d]),
                           peakRob=float(power_profile(inst, Sr).max())))
R["E3"] = E3; log("E3 done", len(E3))

# ---------- E4: static vs rolling re-planning ----------
log("E4 replanning"); E4 = []; RHO = 0.9
def replan_run(wf, d, slack, G, period=6):
    inst, ms = make_inst(wf, slack, F[d], e, G)
    S = robslack(inst, ils_rounds=ILS); D = inst.D
    t = period
    while t < D:
        # reveal actuals for slots < t; shift remaining forecast by decayed residual
        c2 = F[d].copy(); e2 = e.copy()
        c2[:t] = A[d][:t]; e2[:t] = 0.0
        resid = A[d][t - 1] - F[d][t - 1]
        for k in range(t, H): c2[k] = max(40.0, F[d][k] + resid * RHO ** (k - t + 1))
        fixed = {v: S[v] for v in range(inst.n) if S[v] < t}
        inst2 = Instance(inst.dur, inst.pw, inst.proc, inst.edges, inst.D, c2, e2, G, inst.cap)
        S2 = robslack(inst2, release=t, fixed=fixed, ils_rounds=ILS)
        if S2 is not None and check(inst2, S2)[0]: S = S2
        t += period
    return inst, S
for wi, wf in enumerate(WFS[:2]):
    for d in range(0, NDAYS, 2):
        inst, _ = make_inst(wf, 12, F[d], e, 4.0)
        rec = dict(wf=wi, day=d, ASAP=actual_cost(inst, asap(inst), A[d]))
        rec["StaticNom"] = actual_cost(inst, robslack(inst, Gamma=0.0, ils_rounds=ILS), A[d])
        rec["StaticRob4"] = actual_cost(inst, robslack(inst, ils_rounds=ILS), A[d])
        i2, S = replan_run(wf, d, 12, 0.0); assert check(i2, S)[0]; rec["ReplanNom"] = actual_cost(i2, S, A[d])
        i3, S = replan_run(wf, d, 12, 4.0); assert check(i3, S)[0]; rec["ReplanRob4"] = actual_cost(i3, S, A[d])
        E4.append(rec)
R["E4"] = E4; log("E4 done")

# ---------- E5: chain exact algorithm: validation vs MILP, runtime ----------
log("E5 chain"); E5 = []
rng = np.random.default_rng(9)
for n in [5, 8, 10, 12]:
    for rep in range(5):
        dur = list(rng.integers(1, 3, n)); pw = [float(x) for x in rng.choice([1, 2, 3], n)]
        edges = [(i, i + 1) for i in range(n - 1)]
        big = Instance(dur, pw, [0] * n, edges, 500, np.zeros(500), np.zeros(500)); ms = makespan(big, asap(big))
        d = int(rng.integers(0, NDAYS)); inst = Instance(dur, pw, [0] * n, edges, min(H, ms + 12), F[d], e, float(rng.choice([1, 2, 4])))
        t = time.time(); v, S = chain_exact(inst); tc = time.time() - t
        r = solve_milp(inst, time_limit=60)
        E5.append(dict(n=n, D=inst.D, chain=v, milp=r["obj"], t_chain=tc, t_milp=r["time"]))
for n in [25, 50, 100, 200]:   # chain only (MILP too slow); runtime scaling
    dur = [1] * n; pw = [2.0] * n; edges = [(i, i + 1) for i in range(n - 1)]
    cc = np.tile(F[0][:24], 20)[:n * 2 + 20]; ee = np.tile(e[:24], 20)[:n * 2 + 20]
    inst = Instance(dur, pw, [0] * n, edges, n + 20, cc, ee, 3.0)
    t = time.time(); v, S = chain_exact(inst); E5.append(dict(n=n, D=inst.D, chain=v, milp=None, t_chain=time.time() - t, t_milp=None))
R["E5"] = E5; log("E5 done")

# ---------- E6: price of robustness (exact, small instance) ----------
log("E6 price of robustness"); E6 = []
wf = workflow(7, n=10, nproc=3, dmax=2)
for d in range(0, 40, 8):
    for G in range(0, 11):
        inst, _ = make_inst(wf, 8, F[d], e, float(G)); r = solve_milp(inst, time_limit=60)
        E6.append(dict(day=d, Gamma=G, wc=r["obj"], nominal=nominal_cost(inst, r["S"]), actual=actual_cost(inst, r["S"], A[d]), optimal=bool(r["optimal"])))
R["E6"] = E6; log("E6 done")
json.dump(R, open("results.json", "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
log("ALL DONE")
