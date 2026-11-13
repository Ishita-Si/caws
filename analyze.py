import json, numpy as np, matplotlib;

matplotlib.use("Agg");
import matplotlib.pyplot as plt
from scipy import stats

R = json.load(open("results.json"));
out = {};
plt.rcParams.update({"font.size": 9, "figure.dpi": 130})
# ---- E1
E1 = R["E1"];
g = lambda k: np.array([(r[k] - r["opt"]) / r["opt"] for r in E1])
out["E1"] = {k: dict(mean=100 * g(k).mean(), median=100 * np.median(g(k)), max=100 * g(k).max()) for k in
             ["rob", "greedy", "nomopt", "heft", "asap"]}
out["E1"]["n"] = len(E1);
out["E1"]["n_proven_optimal"] = int(sum(r["optimal"] for r in E1))
out["E1"]["rob_hits_opt"] = int((g("rob") < 1e-6).sum());
out["E1"]["milp_time_mean"] = float(np.mean([r["t_milp"] for r in E1]))
# ---- E2
E2 = R["E2"];
GAM = [0, 1, 2, 4, 8];
SLK = [2, 6, 12, 24]


def sel(slack): return [r for r in E2 if r["slack"] == slack]


tab = {}
for s in SLK:
    rs = sel(s);
    a = np.array([r["ASAP"]["actual"] for r in rs]);
    tab[s] = {}
    for m in ["HEFT", "Greedy"] + [f"Rob{G}" for G in GAM]:
        x = np.array([r[m]["actual"] for r in rs])
        tab[s][m] = dict(saving=100 * (1 - x.sum() / a.sum()), worse_than_asap=100 * (x > a + 1e-9).mean())
out["E2_actual"] = tab
# nominal-optimal (Rob0) vs robust: paired tests on actual cost
pt = {}
for s in SLK:
    rs = sel(s);
    n0 = np.array([r["Rob0"]["actual"] for r in rs])
    for G in [2, 4]:
        x = np.array([r[f"Rob{G}"]["actual"] for r in rs]);
        w = stats.wilcoxon(x, n0)
        pt[f"slack{s}_G{G}_vs_Rob0"] = dict(mean_diff_pct=100 * (x.mean() / n0.mean() - 1), p=float(w.pvalue))
    gr = np.array([r["Greedy"]["actual"] for r in rs]);
    x = np.array([r["Rob4"]["actual"] for r in rs]);
    w = stats.wilcoxon(x, gr)
    pt[f"slack{s}_G4_vs_Greedy"] = dict(mean_diff_pct=100 * (x.mean() / gr.mean() - 1), p=float(w.pvalue))
out["E2_tests"] = pt
# fragility: share of days nominal plan emits more than ASAP, vs robust
fr = {s: dict(nominal=100 * np.mean([r["Rob0"]["actual"] > r["ASAP"]["actual"] + 1e-9 for r in sel(s)]),
              robust4=100 * np.mean([r["Rob4"]["actual"] > r["ASAP"]["actual"] + 1e-9 for r in sel(s)])) for s in SLK}
out["fragility"] = fr
# price of robustness (large instances, slack 12): mean nominal cost increase vs worst-case at each evaluation Gamma
rs = sel(12);
por = {}
for G in GAM:
    nom = np.mean([r[f"Rob{G}"]["nominal"] / r["Rob0"]["nominal"] for r in rs]);
    wc = np.mean([r[f"Rob{G}"]["wc"]["8"] / r["Rob0"]["wc"]["8"] for r in rs])
    por[G] = dict(nominal_rel=nom, wc8_rel=wc)
out["price_of_robustness_large"] = por
# worst-case bound coverage: actual <= wc(Gamma) for the plan designed with Gamma
cov = {}
for G in GAM:
    cov[G] = float(np.mean([r[f"Rob{G}"]["actual"] <= r[f"Rob{G}"]["wc"][str(G)] + 1e-9 for r in E2]))
out["bound_coverage"] = cov

# ---- stratified by realised forecast error (post-hoc strata) and tail ratios
from exp_common import data

A_, F_, e_ = data();
dmae = np.abs(A_ - F_).mean(1);
q = np.quantile(dmae, [0.25, 0.5, 0.75])
strat = {}
for s_ in [6, 12]:
    rs = sel(s_);
    ms = ["Greedy", "Rob0", "Rob2", "Rob4", "Rob8"]
    for name, lo, hi in [("Q1 (best forecasts)", -1, q[0]), ("Q2", q[0], q[1]), ("Q3", q[1], q[2]),
                         ("Q4 (worst forecasts)", q[2], 1e9)]:
        sub = [r for r in rs if lo < dmae[r["day"]] <= hi];
        a = sum(r["ASAP"]["actual"] for r in sub)
        strat[f"slack{s_}|{name}"] = {m: 100 * (1 - sum(r[m]["actual"] for r in sub) / a) for m in ms}
out["strata"] = strat
tail = {}
for s_ in SLK:
    rs = sel(s_)
    for m in ["Greedy", "Rob0", "Rob2", "Rob4", "Rob8"]:
        ratio = np.array([r[m]["actual"] / r["ASAP"]["actual"] for r in rs]);
        tail[f"slack{s_}|{m}"] = dict(p90=float(np.quantile(ratio, 0.9)), p99=float(np.quantile(ratio, 0.99)),
                                      max=float(ratio.max()))
out["tail_ratio_vs_asap"] = tail
# figure 1: saving vs slack
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
for m, lab in [("HEFT", "HEFT-like"), ("Greedy", "Greedy window"), ("Rob0", "Nominal-opt (Γ=0)"),
               ("Rob2", "RobSlack Γ=2"), ("Rob4", "RobSlack Γ=4"), ("Rob8", "RobSlack Γ=8")]:
    ax[0].plot(SLK, [tab[s][m]["saving"] for s in SLK], marker="o", label=lab)
    ax[1].plot(SLK, [tab[s][m]["worse_than_asap"] for s in SLK], marker="o", label=lab)
ax[0].set_xlabel("deadline slack (slots)");
ax[0].set_ylabel("saving vs ASAP on actual (%)");
ax[1].set_xlabel("deadline slack (slots)");
ax[1].set_ylabel("days worse than ASAP (%)")
ax[0].legend(fontsize=6);
plt.tight_layout();
plt.savefig("fig_slack.png")
# ---- E3
E3 = R["E3"];
rhos = [None, 0.9, 0.75, 0.6, 0.5];
c3 = {}
for rho in rhos:
    rs = [r for r in E3 if r["rho"] == rho]
    if not rs: continue
    a = sum(r["ASAP"] for r in rs);
    c3[str(rho)] = dict(rob=100 * (1 - sum(r["Rob4"] for r in rs) / a), nom=100 * (1 - sum(r["Nom"] for r in rs) / a),
                        n=len(rs))
out["E3"] = c3
fig, ax = plt.subplots(figsize=(3.6, 2.8));
ks = list(c3)
ax.plot(range(len(ks)), [c3[k]["rob"] for k in ks], marker="o", label="RobSlack Γ=4");
ax.plot(range(len(ks)), [c3[k]["nom"] for k in ks], marker="s", label="Nominal-opt")
ax.set_xticks(range(len(ks)));
ax.set_xticklabels(["none" if k == "None" else k for k in ks]);
ax.set_xlabel("cap / ASAP peak");
ax.set_ylabel("saving vs ASAP (%)");
ax.legend(fontsize=7);
plt.tight_layout();
plt.savefig("fig_cap.png")
# ---- E4
E4 = R["E4"];
a = sum(r["ASAP"] for r in E4)
out["E4"] = {k: 100 * (1 - sum(r[k] for r in E4) / a) for k in ["StaticNom", "StaticRob4", "ReplanNom", "ReplanRob4"]}
out["E4"]["n"] = len(E4)
out["E4_tests"] = {f"{k}_vs_StaticRob4": float(stats.wilcoxon([r[k] for r in E4], [r["StaticRob4"] for r in E4]).pvalue)
                   for k in ["StaticNom", "ReplanNom", "ReplanRob4"]}
# ---- E5
E5 = R["E5"];
v = [r for r in E5 if r["milp"] is not None]
out["E5"] = dict(max_rel_diff=max(abs(r["chain"] - r["milp"]) / r["milp"] for r in v), n=len(v),
                 runtime=[(r["n"], r["t_chain"], r["t_milp"]) for r in E5 if r["n"] in (10, 12, 25, 50, 100, 200)])
# ---- E6
E6 = R["E6"];
fig, ax = plt.subplots(1, 2, figsize=(7, 2.8));
days = sorted({r["day"] for r in E6});
tot = {}
for G in range(11):
    rs = [r for r in E6 if r["Gamma"] == G];
    tot[G] = (np.mean([r["wc"] for r in rs]), np.mean([r["nominal"] for r in rs]), np.mean([r["actual"] for r in rs]))
Gs = list(tot);
ax[0].plot(Gs, [tot[g][0] / tot[0][0] for g in Gs], marker="o", label="optimal worst-case value (concave)")
ax[0].set_xlabel("Γ");
ax[0].set_ylabel("rel. to Γ=0");
ax[0].legend(fontsize=7)
for d in days:
    rs = sorted([r for r in E6 if r["day"] == d], key=lambda r: r["Gamma"])
    ax[1].plot([r["nominal"] / rs[0]["nominal"] for r in rs], [r["Gamma"] for r in rs], alpha=0.5)
ax[1].set_xlabel("forecast cost of the plan, relative to Γ=0 plan");
ax[1].set_ylabel("Γ used in planning");
plt.tight_layout();
plt.savefig("fig_price.png")
out["E6"] = {str(G): dict(wc=tot[G][0], nominal=tot[G][1], actual=tot[G][2]) for G in Gs}
out["E6_nominal_increase_pct_at_G4"] = 100 * (tot[4][1] / tot[0][1] - 1)
json.dump(out, open("summary.json", "w"), indent=1, default=float);
print(json.dumps(out, indent=1, default=float))
