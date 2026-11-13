"""Core model: instance, exact worst-case cost, independent checker, brute force, ASAP/HEFT-style baselines."""
from dataclasses import dataclass, field
import numpy as np, itertools

@dataclass
class Instance:
    dur: list            # duration (slots) per task
    pw: list             # power (kW) per task
    proc: list           # processor id per task
    edges: list          # (u,v): u before v
    D: int               # deadline: all tasks end <= D (slots 0..D-1)
    c: np.ndarray        # forecast carbon per slot (length >= D)
    e: np.ndarray        # max error per slot
    Gamma: float = 0.0
    cap: float = np.inf
    h: float = 1.0
    @property
    def n(self): return len(self.dur)
    def preds(self):
        p = [[] for _ in range(self.n)]
        for u, v in self.edges: p[v].append(u)
        return p
    def succs(self):
        s = [[] for _ in range(self.n)]
        for u, v in self.edges: s[u].append(v)
        return s
    def topo(self):
        pr = self.preds(); sc = self.succs(); indeg = [len(x) for x in pr]
        q = [v for v in range(self.n) if indeg[v] == 0]; out = []
        while q:
            v = q.pop(0); out.append(v)
            for w in sc[v]:
                indeg[w] -= 1
                if indeg[w] == 0: q.append(w)
        assert len(out) == self.n, "cycle"
        return out

def power_profile(inst, S):
    P = np.zeros(inst.D)
    for v, s in enumerate(S):
        P[s:s + inst.dur[v]] += inst.pw[v]
    return P

def topgamma(x, Gamma):
    """sum of the Gamma largest entries of x>=0, fractional part applied to next largest."""
    k = int(np.floor(Gamma + 1e-12)); fr = Gamma - k
    xs = np.sort(x)[::-1]
    tot = xs[:k].sum()
    if fr > 1e-12 and k < len(xs): tot += fr * xs[k]
    return tot

def nominal_cost(inst, S, c=None):
    c = inst.c if c is None else c
    return inst.h * float(c[:inst.D] @ power_profile(inst, S))

def worst_case_cost(inst, S, Gamma=None):
    G = inst.Gamma if Gamma is None else Gamma
    P = power_profile(inst, S)
    return inst.h * (float(inst.c[:inst.D] @ P) + topgamma(inst.e[:inst.D] * P, G))

def actual_cost(inst, S, a):
    return inst.h * float(np.asarray(a)[:inst.D] @ power_profile(inst, S))

def check(inst, S, tol=1e-9):
    """Independent feasibility checker, recomputed from scratch. Returns (ok, message)."""
    if len(S) != inst.n: return False, "length"
    for v, s in enumerate(S):
        if s < 0 or s + inst.dur[v] > inst.D: return False, f"task {v} outside [0,D]"
    for u, v in inst.edges:
        if S[u] + inst.dur[u] > S[v]: return False, f"precedence {u}->{v}"
    occ = {}
    for v, s in enumerate(S):
        for t in range(s, s + inst.dur[v]):
            key = (inst.proc[v], t)
            if key in occ: return False, f"processor overlap {occ[key]},{v} at {t}"
            occ[key] = v
    tot = {}
    for v, s in enumerate(S):
        for t in range(s, s + inst.dur[v]): tot[t] = tot.get(t, 0) + inst.pw[v]
    for t, p in tot.items():
        if p > inst.cap + tol: return False, f"cap violated at {t}"
    return True, "ok"

def worst_case_bruteforce(inst, S, Gamma=None):
    """Adversary by enumeration (integer Gamma): pick best subset of <=Gamma slots. Independent of topgamma."""
    G = inst.Gamma if Gamma is None else Gamma
    assert abs(G - round(G)) < 1e-12
    P = power_profile(inst, S); base = inst.c[:inst.D] @ P
    used = [t for t in range(inst.D) if P[t] > 0]
    best = 0.0
    for k in range(0, min(int(round(G)), len(used)) + 1):
        for sub in itertools.combinations(used, k):
            best = max(best, sum(inst.e[t] * P[t] for t in sub))
    return inst.h * (base + best)

# ---------- baselines ----------
def upward_rank(inst):
    sc = inst.succs(); r = [0] * inst.n
    for v in reversed(inst.topo()):
        r[v] = inst.dur[v] + max([r[w] for w in sc[v]], default=0)
    return r

def list_schedule(inst, order, release=0, fixed=None):
    """Serial schedule generation: place tasks in `order` at earliest feasible start (prec, processor, cap)."""
    pr = inst.preds(); S = [None] * inst.n
    P = np.zeros(inst.D + 1); busy = {}
    fixed = fixed or {}
    for v, s in fixed.items():
        S[v] = s; P[s:s + inst.dur[v]] += inst.pw[v]
        for t in range(s, s + inst.dur[v]): busy[(inst.proc[v], t)] = True
    for v in order:
        if S[v] is not None: continue
        est = max([release] + [S[u] + inst.dur[u] for u in pr[v]])
        s = est
        while True:
            if s + inst.dur[v] > inst.D: return None
            if all((inst.proc[v], t) not in busy for t in range(s, s + inst.dur[v])) and \
               np.all(P[s:s + inst.dur[v]] + inst.pw[v] <= inst.cap + 1e-9):
                break
            s += 1
        S[v] = s; P[s:s + inst.dur[v]] += inst.pw[v]
        for t in range(s, s + inst.dur[v]): busy[(inst.proc[v], t)] = True
    return S

def asap(inst, release=0, fixed=None):
    return list_schedule(inst, inst.topo(), release, fixed)

def heft_like(inst, release=0, fixed=None):
    """Carbon-agnostic: upward-rank priority list scheduling (processor mapping is given, so HEFT reduces to this)."""
    r = upward_rank(inst); topo = inst.topo(); pos = {v: i for i, v in enumerate(topo)}
    order = sorted(range(inst.n), key=lambda v: (-r[v], pos[v]))
    # rank order is a valid topological order since dur>0
    return list_schedule(inst, order, release, fixed)

def makespan(inst, S): return max(S[v] + inst.dur[v] for v in range(inst.n))

def brute_force_opt(inst, objective="worst"):
    """Enumerate all feasible schedules (tiny instances). Returns (best value, best S)."""
    n = inst.n; topo = inst.topo(); pr = inst.preds(); best = (np.inf, None)
    def rec(i, S, P, busy):
        nonlocal best
        if i == n:
            val = worst_case_cost(inst, S) if objective == "worst" else nominal_cost(inst, S)
            if val < best[0] - 1e-12: best = (val, list(S))
            return
        v = topo[i]
        est = max([0] + [S[u] + inst.dur[u] for u in pr[v]])
        for s in range(est, inst.D - inst.dur[v] + 1):
            ts = range(s, s + inst.dur[v])
            if any((inst.proc[v], t) in busy for t in ts): continue
            if any(P[t] + inst.pw[v] > inst.cap + 1e-9 for t in ts): continue
            S[v] = s
            for t in ts: P[t] += inst.pw[v]; busy.add((inst.proc[v], t))
            rec(i + 1, S, P, busy)
            for t in ts: P[t] -= inst.pw[v]; busy.discard((inst.proc[v], t))
            S[v] = None
    rec(0, [None] * n, np.zeros(inst.D + 1), set())
    return best
