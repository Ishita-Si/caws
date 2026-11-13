"""Quick check: is this dataset likely to show a benefit from robust scheduling?
Usage: python screen_dataset.py your_data.csv      (columns: timestamp,actual,forecast; hourly)"""
import sys, pandas as pd, numpy as np
df = pd.read_csv(sys.argv[1], parse_dates=["timestamp"])
err = (df.actual - df.forecast).abs()
rng = df.groupby(df.timestamp.dt.date).actual.agg(lambda x: x.max() - x.min()).mean()
days = len(df) // 24
print(f"rows {len(df)} (~{days} days) | mean actual {df.actual.mean():.1f} | mean daily range {rng:.1f}")
print(f"MAE {err.mean():.1f} | p90 error {err.quantile(.9):.1f} | bias {(df.actual-df.forecast).mean():.1f}")
r = err.mean() / rng
print(f"error / daily range = {r:.2f}   (UK data: 0.09; our scaled tests: about 0.3 at k=3, 0.55 at k=6)")
print("verdict:", "robustness likely to matter" if r > 0.25 else "errors small: expect nominal ~ robust")
if days < 100: print("WARNING: fewer than 100 days; need >=60 train + 40 test")
