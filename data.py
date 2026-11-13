"""Synthetic day-ahead carbon-intensity generator with realistic forecast error structure, plus a real-data CSV loader.
(No real grid data is reachable from the sandbox; see README. Replace `load_csv` output into the same arrays.)"""
import numpy as np
H = 48  # planning horizon (slots/hours): two days, forecast issued at midnight of day 0
def make_series(n_days, seed=0, phi=0.97):
    """returns actual[n_days,H], forecast[n_days,H]  (g CO2/kWh)"""
    rng = np.random.default_rng(seed); total = (n_days + 2) * 24
    hrs = np.arange(total) % 24; day = np.arange(total) // 24
    # cloudiness per day (AR across days): solar availability factor in [0.15, 1]
    s = np.zeros(n_days + 2); s[0] = 0.6
    for d in range(1, len(s)): s[d] = np.clip(0.6 + 0.6 * (s[d - 1] - 0.6) + rng.normal(0, 0.25), 0.15, 1.0)
    solar_shape = np.maximum(0, np.sin(np.pi * (hrs - 6) / 12)) ** 1.5
    demand = 60 * np.cos(2 * np.pi * (hrs - 19) / 24)          # evening peak => fossil on the margin
    wind = np.zeros(total)
    for t in range(1, total): wind[t] = phi * wind[t - 1] + rng.normal(0, 14)
    base = 340 + demand
    actual = base - 190 * s[day] * solar_shape - 1.3 * wind + rng.normal(0, 8, total)
    actual = np.clip(actual, 40, 700)
    fc_all = []
    A = np.zeros((n_days, H)); F = np.zeros((n_days, H))
    for d in range(1, n_days + 1):
        t0 = d * 24; k = np.arange(H); sl = slice(t0, t0 + H)
        issue = t0 - 1   # information up to the last hour before the horizon
        w0 = wind[issue]
        f = base[sl] - 190 * 0.6 * solar_shape[sl] - 1.3 * w0 * phi ** (k + 1)  # mean cloudiness, wind persistence
        A[d - 1] = actual[sl]; F[d - 1] = np.clip(f, 40, 700)
    return A, F

def calibrate(A_train, F_train, q=0.9, smooth=3):
    """per-slot max error = q-quantile of |actual - forecast| on training days, lightly smoothed over lead time"""
    err = np.quantile(np.abs(A_train - F_train), q, axis=0)
    k = np.ones(smooth) / smooth
    pad = np.concatenate([err[:smooth // 2 + 1][::-1][-(smooth // 2):] if smooth > 1 else [], err, err[-(smooth // 2):] if smooth > 1 else []])
    return np.convolve(pad, k, mode="valid")[:len(err)] if smooth > 1 else err

def load_csv(path):
    """Expect columns: timestamp,actual,forecast (hourly, forecast = day-ahead). Returns (A,F) arrays of shape [days,H]."""
    import csv
    rows = list(csv.DictReader(open(path))); a = np.array([float(r["actual"]) for r in rows]); f = np.array([float(r["forecast"]) for r in rows])
    nd = (len(a) - H) // 24
    return np.stack([a[d * 24:d * 24 + H] for d in range(nd)]), np.stack([f[d * 24:d * 24 + H] for d in range(nd)])
