"""Aggregate per-fold results, run significance tests, and draw manuscript figures (revision R3)."""
import os, json
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIG = os.path.join(R, "figures"); os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#222222", "#8a8a8a"
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "lines.linewidth": 2})


def corrected_t(a, b, test_frac, k=15):
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated k-fold CV."""
    d = np.asarray(a) - np.asarray(b); J = len(d)
    if np.allclose(d, 0): return 0.0, 1.0
    var = np.var(d, ddof=1) * (1 / J + test_frac / (1 - test_frac))
    t = d.mean() / np.sqrt(var); p = 2 * stats.t.sf(abs(t), J - 1)
    return t, p


def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1, run)
    return adj


def summarise(df, keys, metrics_):
    g = df.groupby(keys)
    out = g[metrics_].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["n_folds"] = g.size()
    return out.reset_index()


def pairwise(df, keys_fixed, ref, others, metric, test_frac=0.2):
    rows = []
    for o in others:
        a = df[df.model == ref].sort_values("fold")[metric].values
        b = df[df.model == o].sort_values("fold")[metric].values
        if len(a) != len(b) or len(a) == 0: continue
        t, p = corrected_t(a, b, test_frac)
        try: w = stats.wilcoxon(a, b).pvalue if not np.allclose(a, b) else 1.0
        except ValueError: w = 1.0
        rows.append(dict(**keys_fixed, ref=ref, other=o, metric=metric, mean_diff=float(np.mean(a - b)),
                         sd_diff=float(np.std(a - b, ddof=1)), t_corr=t, p_corr=p, p_wilcoxon=w))
    if rows:
        adj = holm([r["p_corr"] for r in rows])
        for r, a_ in zip(rows, adj): r["p_corr_holm"] = a_
    return rows


report = {}
# ------------------------------------------------------------------ Windows
if os.path.exists(os.path.join(R, "windows_folds.csv")):
    w = pd.read_csv(os.path.join(R, "windows_folds.csv"))
    mets = ["acc", "bal_acc", "macro_f1", "weighted_f1", "auc", "mcc", "train_acc"]
    s = summarise(w, ["feat", "task", "model"], [m for m in mets if m in w])
    s.to_csv(os.path.join(R, "windows_summary.csv"), index=False)
    tests = []
    for (feat, task), d in w[w.feat == "clean"].groupby(["feat", "task"]):
        for metric in ["acc", "macro_f1", "auc"]:
            tests += pairwise(d, {"feat": feat, "task": task}, "GAT",
                              ["Majority", "LogReg", "RandomForest", "XGBoost", "MLP(no graph)", "GCN"], metric)
            tests += pairwise(d, {"feat": feat, "task": task}, "GCN", ["MLP(no graph)"], metric)
    pd.DataFrame(tests).to_csv(os.path.join(R, "windows_significance.csv"), index=False)
    gd = w[w.model.isin(["GCN", "GAT", "MLP(no graph)"])]
    ov = gd.groupby(["feat", "task", "model"]).agg(train_acc=("train_acc", "mean"), val_acc=("val_acc", "mean"),
                                                   test_acc=("acc", "mean"), best_epoch=("best_epoch", "mean"),
                                                   epochs_run=("epochs_run", "mean")).reset_index()
    ov["train_test_gap"] = ov.train_acc - ov.test_acc
    ov.to_csv(os.path.join(R, "windows_overfitting.csv"), index=False)

    # Learning curves (fold 0, clean multiclass) for GCN and GAT
    cpath = os.path.join(R, "windows_curves_fold0.json")
    if os.path.exists(cpath):
        cur = json.load(open(cpath))
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
        for ax, nm in zip(axes, ["GCN", "GAT"]):
            h = np.array(cur.get(f"clean_multiclass_{nm}", []))
            if len(h) == 0: continue
            ax.plot(h[:, 0], h[:, 3], color=BLUE, label="Training accuracy")
            ax.plot(h[:, 0], h[:, 4], color=ORANGE, linestyle="--", label="Validation accuracy")
            ax.axhline(0.8334, color=MUTED, linewidth=1, linestyle=":")
            ax.text(h[-1, 0], 0.8334, " majority rate", va="bottom", ha="right", color=INK, fontsize=8)
            ax.set_title(f"{nm} (5-class, leakage-free features)", fontsize=10)
            ax.set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy"); axes[0].legend(frameon=False, fontsize=8, loc="lower right")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_windows_curves.png"), dpi=300); plt.close(fig)

    pt = os.path.join(R, "windows_permutation_test.json")
    if os.path.exists(pt):
        report["windows_permutation"] = json.load(open(pt))

    ra = os.path.join(R, "windows_resample_ablation.csv")
    if os.path.exists(ra):
        rdf = pd.read_csv(ra)
        _g = rdf.groupby("model")[["acc", "macro_f1"]].agg(["mean", "std"]).round(4)
        _g.columns = [f"{m}_{s_}" for m, s_ in _g.columns]
        report["windows_resample_ablation"] = _g.to_dict(orient="index")

# ------------------------------------------------------------------ Android
import glob
_af = sorted(glob.glob(os.path.join(R, "android_folds_*.csv")))
if _af:
    a = pd.concat([pd.read_csv(f) for f in _af], ignore_index=True)
    mets = ["acc", "bal_acc", "precision", "recall", "f1", "macro_f1", "auc", "mcc", "train_acc"]
    s = summarise(a, ["protocol", "feat", "model"], [m for m in mets if m in a])
    s.to_csv(os.path.join(R, "android_summary.csv"), index=False)
    tests = []
    for (proto, feat), d in a.groupby(["protocol", "feat"]):
        models = d.model.unique().tolist()
        for ref in ["FT-Transformer", "TabTransformer(single-token, original)"]:
            if ref not in models: continue
            for metric in ["acc", "f1", "auc"]:
                tests += pairwise(d, {"protocol": proto, "feat": feat}, ref, [m for m in models if m != ref], metric)
    pd.DataFrame(tests).to_csv(os.path.join(R, "android_significance.csv"), index=False)

    # Random vs duplicate-aware grouped CV, all 169 features
    sel = ["LogReg", "SVM-RBF", "RandomForest", "XGBoost", "MLP", "TabTransformer(single-token, original)", "FT-Transformer"]
    lab = ["LogReg", "SVM", "RF", "XGBoost", "MLP", "TabTr.\n(orig.)", "FT-Tr."]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    xs = np.arange(len(sel))
    for off, proto, col, name in [(-0.18, "random", BLUE, "Random stratified CV"),
                                  (0.18, "dedup_grouped", ORANGE, "Duplicate-aware grouped CV")]:
        d = a[(a.protocol == proto) & (a.feat == "all169")]
        m = [d[d.model == s_].f1.mean() for s_ in sel]; sd = [d[d.model == s_].f1.std() for s_ in sel]
        ax.bar(xs + off, m, width=0.34, color=col, yerr=sd, capsize=2, error_kw={"elinewidth": 1, "ecolor": INK},
               label=name, edgecolor="white", linewidth=1)
    ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=8); ax.set_ylabel("Malware-class F1 (mean ± SD)")
    lo = a[a.feat == "all169"].f1.min(); ax.set_ylim(max(0, lo - 0.08), 1.0)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_android_protocols.png"), dpi=300); plt.close(fig)
    da = os.path.join(R, "android_duplicate_audit.json")
    if os.path.exists(da): report["android_duplicates"] = json.load(open(da))

# Permutation-test figure needs the null distribution; recompute summary only
json.dump(report, open(os.path.join(R, "report.json"), "w"), indent=1, default=str)
print(json.dumps(report, indent=1, default=str))
