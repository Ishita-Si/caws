"""3-Partition -> carbon scheduling reduction (Gamma=0, one processor, no precedence) and numerical check."""
import numpy as np, itertools, random
from model import *; from milp import solve_milp

def reduction(a, B):
    m = len(a) // 3; D = m * B + (m - 1)
    c = np.zeros(D)
    for j in range(1, m): c[j * (B + 1) - 1] = 1.0   # brown separator slots
    return Instance(list(a), [1.0] * len(a), [0] * len(a), [], D, c, np.zeros(D), 0.0, np.inf)

def has_3partition(a, B):
    a = sorted(a, reverse=True); m = len(a) // 3
    def rec(rem, bins):
        if not rem: return True
        x = rem[0]; seen = set()
        for i in range(m):
            if bins[i][0] + x <= B and (bins[i][0], bins[i][1]) not in seen:
                seen.add((bins[i][0], bins[i][1]))
                if bins[i][1] >= 3: continue
                nb = list(bins); nb[i] = (bins[i][0] + x, bins[i][1] + 1)
                if rec(rem[1:], nb): return True
        return False
    return sum(a) == m * B and rec(a, [(0, 0)] * m)

def run(trials=60, seed=0):
    rnd = random.Random(seed); agree = yes = no = 0
    for _ in range(trials):
        m = rnd.choice([2, 3]); B = rnd.choice([13, 17, 19, 23])
        # item sizes strictly in (B/4, B/2)
        lo, hi = B // 4 + 1, (B - 1) // 2
        def planted():
            a = []
            for _ in range(m):
                while True:
                    x, y = rnd.randint(lo, hi), rnd.randint(lo, hi); z = B - x - y
                    if lo <= z <= hi: a += [x, y, z]; break
            return a
        a = planted()
        if rnd.random() < 0.8:   # perturb to (usually) break 3-partitionability, keep sum and size range
            for _ in range(60):
                i, j = rnd.sample(range(3 * m), 2); k = rnd.randint(1, 2)
                if lo <= a[i] + k <= hi and lo <= a[j] - k <= hi: a[i] += k; a[j] -= k
        inst = reduction(a, B); r = solve_milp(inst, time_limit=60)
        zero = r["obj"] is not None and r["obj"] < 1e-6
        truth = has_3partition(a, B)
        agree += (zero == truth); yes += truth; no += (not truth)
    return agree, trials, yes, no
if __name__ == "__main__":
    print(run())
