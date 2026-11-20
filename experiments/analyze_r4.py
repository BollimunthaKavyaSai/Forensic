"""Aggregate R4 results (KronoDroid, ATLASv2, CFReDS, F-Droid), run significance tests and draw figures."""
import os, json, glob
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def corrected_t(a, b, test_frac):
    """Nadeau & Bengio (2003) corrected resampled t-test."""
    d = np.asarray(a) - np.asarray(b); J = len(d)
    if np.allclose(d, 0): return 0.0, 1.0
    t = d.mean() / np.sqrt(np.var(d, ddof=1) * (1 / J + test_frac / (1 - test_frac)))
    return t, 2 * stats.t.sf(abs(t), J - 1)


def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1, run)
    return adj

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); FIG = os.path.join(R, "figures")
BLUE, ORANGE, AQUA, INK, MUTED, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#222222", "#8a8a8a", "#7b5ea7"
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.grid": True, "axes.axisbelow": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "lines.linewidth": 2})
rep = {}

# ------------------------------------------------------------------ KronoDroid
kf = [f for f in glob.glob(os.path.join(R, "kronodroid_folds_*.csv"))]
if kf:
    k = pd.concat([pd.read_csv(f) for f in kf], ignore_index=True)
    k = k[k.model != "FOLD_DONE"].drop_duplicates(["protocol", "feat", "fold", "model"], keep="last")
    mets = ["acc", "bal_acc", "f1", "auc", "mcc", "precision", "recall"]
    g = k.groupby(["protocol", "feat", "model"])
    s = g[[m for m in mets if m in k]].agg(["mean", "std"]); s.columns = [f"{a}_{b}" for a, b in s.columns]
    s["n_folds"] = g.size(); s = s.reset_index(); s.to_csv(os.path.join(R, "kronodroid_summary.csv"), index=False)
    tests = []
    for (proto, feat), d in k.groupby(["protocol", "feat"]):
        models = [m for m in d.model.unique() if m != "Majority"]
        for ref in ["FT-Transformer", "RandomForest", "XGBoost"]:
            if ref not in models: continue
            for metric in ["f1", "auc"]:
                rows = []
                for o in models:
                    if o == ref: continue
                    a = d[d.model == ref].set_index("fold")[metric]; b = d[d.model == o].set_index("fold")[metric]
                    common = a.index.intersection(b.index)
                    if len(common) < 3: continue
                    tf = 0.2 if proto != "temporal" else None
                    if tf:
                        t, p = corrected_t(a[common].values, b[common].values, tf)
                    else:  # temporal folds are disjoint test years: ordinary paired t-test
                        t, p = stats.ttest_rel(a[common].values, b[common].values)
                    try: w = stats.wilcoxon(a[common].values, b[common].values).pvalue
                    except ValueError: w = 1.0
                    rows.append(dict(protocol=proto, feat=feat, ref=ref, other=o, metric=metric, n=len(common),
                                     mean_diff=float((a[common] - b[common]).mean()), t=t, p=p, p_wilcoxon=w))
                if rows:
                    for r_, ad in zip(rows, holm([r["p"] for r in rows])): r_["p_holm"] = ad
                tests += rows
    pd.DataFrame(tests).to_csv(os.path.join(R, "kronodroid_significance.csv"), index=False)

    # Figure: F1 and ROC-AUC by protocol (all features)
    sel = ["LogReg", "RandomForest", "XGBoost", "MLP", "FT-Transformer"]; lab = ["LogReg", "RF", "XGBoost", "MLP", "FT-Tr."]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, metric, ttl in zip(axes, ["f1", "auc"], ["Malware-class F1", "ROC-AUC"]):
        xs = np.arange(len(sel))
        for off, proto, col, name in [(-0.27, "random", BLUE, "Random CV"), (0, "family", ORANGE, "Unseen families"),
                                      (0.27, "temporal", AQUA, "Future years")]:
            d = k[(k.protocol == proto) & (k.feat == "all")]
            m = [d[d.model == s_][metric].mean() for s_ in sel]; sd = [d[d.model == s_][metric].std() for s_ in sel]
            ax.bar(xs + off, m, 0.25, color=col, yerr=sd, capsize=2, error_kw={"elinewidth": 0.8, "ecolor": INK},
                   label=name, edgecolor="white", linewidth=0.8)
        ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=8); ax.set_title(ttl, fontsize=10)
        lo = np.nanmin([k[(k.feat == "all") & (k.model.isin(sel))][metric].quantile(0.02)]); ax.set_ylim(max(0, lo - 0.08), 1.0)
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_kd_protocols.png"), dpi=300); plt.close(fig)

    # Figure: temporal decay
    d = k[(k.protocol == "temporal") & (k.feat == "all")].copy(); d["year"] = d.fold.str[1:].astype(int)
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    for nm, col, ls in [("RandomForest", BLUE, "-"), ("XGBoost", AQUA, "-"), ("LogReg", MUTED, ":"), ("MLP", PURPLE, "--"),
                        ("FT-Transformer", ORANGE, "-.")]:
        q = d[d.model == nm].sort_values("year")
        if len(q): ax.plot(q.year, q.auc, ls, color=col, marker="o", markersize=3, label=nm)
    ax.set_xlabel("Test year (model trained on all earlier years)"); ax.set_ylabel("ROC-AUC")
    ax.legend(frameon=False, fontsize=7.5, ncol=5, loc="lower left"); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_kd_temporal.png"), dpi=300); plt.close(fig)

# ------------------------------------------------------------------ ATLASv2
af = glob.glob(os.path.join(R, "atlas_folds_*.csv"))
if af:
    a = pd.concat([pd.read_csv(f) for f in af], ignore_index=True)
    if "featset" not in a: a["featset"] = "full"
    a["featset"] = a.featset.fillna("full")
    per = a.groupby(["featset", "protocol", "model", "test_scenario"])[["auc", "ap", "f1", "recall_at_5pct",
                                                                        "effort_full_recall"]].mean().reset_index()
    g = per.groupby(["featset", "protocol", "model"])
    s = g[["auc", "ap", "f1", "recall_at_5pct", "effort_full_recall"]].agg(["mean", "std"])
    s.columns = [f"{x}_{y}" for x, y in s.columns]; s["n_scenarios"] = g.size(); s = s.reset_index()
    s.to_csv(os.path.join(R, "atlas_summary.csv"), index=False)
    tests = []
    for (fs, proto), d in per.groupby(["featset", "protocol"]):
        tab = ["LogReg", "RandomForest", "XGBoost", "MLP (no graph)"]
        best_tab = max(tab, key=lambda m: d[d.model == m].ap.mean())
        for metric in ["ap", "auc", "f1"]:
            rows = []
            for gm in ["GCN (provenance)", "GraphSAGE (provenance)", "GAT (provenance)", "GCN (k-NN graph)"]:
                x = d[d.model == gm].set_index("test_scenario")[metric]; y = d[d.model == best_tab].set_index("test_scenario")[metric]
                c = x.index.intersection(y.index)
                if len(c) < 3: continue
                diff = (x[c] - y[c]).values
                try: w = stats.wilcoxon(x[c], y[c]).pvalue
                except ValueError: w = 1.0
                rng = np.random.RandomState(0); bs = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
                rows.append(dict(featset=fs, protocol=proto, graph_model=gm, best_tabular=best_tab, metric=metric,
                                 n=len(c), mean_diff=float(diff.mean()), ci_lo=float(np.percentile(bs, 2.5)),
                                 ci_hi=float(np.percentile(bs, 97.5)), p_wilcoxon=w))
            if rows:
                for r_, ad in zip(rows, holm([r["p_wilcoxon"] for r in rows])): r_["p_holm"] = ad
            tests += rows
    pd.DataFrame(tests).to_csv(os.path.join(R, "atlas_significance.csv"), index=False)

    d = per[(per.featset == "full") & (per.protocol == "cross_vector")]
    order = ["LogReg", "RandomForest", "XGBoost", "MLP (no graph)", "GCN (k-NN graph)", "GCN (provenance)",
             "GraphSAGE (provenance)", "GAT (provenance)"]
    labs = ["LogReg", "RF", "XGBoost", "MLP", "GCN\nk-NN", "GCN\nprov.", "SAGE\nprov.", "GAT\nprov."]
    cols = [BLUE] * 4 + [MUTED] + [ORANGE] * 3
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, metric, ttl in zip(axes, ["ap", "effort_full_recall"], ["Average precision (higher is better)",
                                                                  "Share of processes reviewed to find all attack\nprocesses (lower is better)"]):
        m = [d[d.model == o][metric].mean() for o in order]; sd = [d[d.model == o][metric].std() for o in order]
        ax.bar(range(len(order)), m, color=cols, yerr=sd, capsize=2, error_kw={"elinewidth": 0.8, "ecolor": INK},
               edgecolor="white")
        ax.set_xticks(range(len(order))); ax.set_xticklabels(labs, fontsize=7); ax.set_title(ttl, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_atlas_models.png"), dpi=300); plt.close(fig)

# ------------------------------------------------------------------ CFReDS + F-Droid
for nm in ["cfreds_validation_as_released", "cfreds_validation_fixed", "android_fdroid_e2e"]:
    p = os.path.join(R, nm + ".json")
    if os.path.exists(p): rep[nm] = json.load(open(p))
if "cfreds_validation_fixed" in rep:
    sc = rep["cfreds_validation_fixed"]["scores"]; sc0 = rep.get("cfreds_validation_as_released", {}).get("scores", {})
    cats = [("System identity\n(name, TZ, last user)", ["computer_name", "timezone", "last_logged_user"]),
            ("Network\nsettings", ["ip_address", "dhcp_server", "default_gateway"]),
            ("Installed\napplications", ["installed_apps"]), ("USB device\nserials", ["usb_serials"]),
            ("Executed\nprograms", ["executed_programs"]), ("User\nprofiles", ["user_profiles"]),
            ("Browser search\nkeywords", ["search_keywords"]), ("Logon/startup\ntimeline (.evtx)", ["evtx_logon_timeline"])]
    rows = []
    for lab_, keys in cats:
        f1_ = sum(sc[k_]["found"] for k_ in keys); t1 = sum(sc[k_]["total"] for k_ in keys)
        f0 = sum(sc0.get(k_, {}).get("found", 0) for k_ in keys)
        rows.append(dict(category=lab_.replace("\n", " "), found_released=f0, found_fixed=f1_, total=t1))
    cdf = pd.DataFrame(rows); cdf.to_csv(os.path.join(R, "cfreds_scores_by_category.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7.2, 2.8)); xs = np.arange(len(cats))
    ax.bar(xs, cdf.found_fixed / cdf.total, 0.6, color=BLUE)
    for x, (f_, t_) in enumerate(zip(cdf.found_fixed, cdf.total)): ax.text(x, f_ / t_ + 0.02, f"{f_}/{t_}", ha="center", fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels([c[0] for c in cats], fontsize=7); ax.set_ylim(0, 1.1)
    ax.set_ylabel("Share of answer-key items")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_cfreds_recall.png"), dpi=300); plt.close(fig)
    rep["cfreds_totals"] = dict(found_fixed=int(cdf.found_fixed.sum()), found_released=int(cdf.found_released.sum()),
                                total=int(cdf.total.sum()))
json.dump(rep, open(os.path.join(R, "report_r4.json"), "w"), indent=1, default=str)
print(json.dumps({k_: v for k_, v in rep.items() if k_ == "cfreds_totals" or k_ == "android_fdroid_e2e"}, indent=1, default=str))
