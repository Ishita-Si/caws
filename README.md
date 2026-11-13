# Robust Carbon-Aware Workflow Scheduling — code & reproducibility
Files: model.py (instance, exact worst-case cost, independent checker, brute force, ASAP/HEFT-like), milp.py (exact robust MILP, HiGHS),
chain.py (exact theta-threshold DP, single processor fixed order), hardness.py (3-Partition reduction + check), heur.py (RobSlack),
data.py (synthetic day-ahead carbon generator + load_csv for real data), gen.py/exp_common.py (instances), test_all.py, test_heur.py,
run_experiments.py (E1-E6 -> results.json), run_e7.py (-> e7.json), analyze.py (-> summary.json, fig_*.png).
Run: python3 test_all.py; python3 hardness.py; python3 run_experiments.py; python3 analyze.py; python3 run_e7.py   (needs numpy, scipy>=1.9, matplotlib)
IMPORTANT: carbon data are SYNTHETIC (sandbox had no access to grid data). To use real data: data.load_csv(path) with columns timestamp,actual,forecast,
then re-run. Everything else is unchanged. All reported numbers must be regenerated on real data before submission.
