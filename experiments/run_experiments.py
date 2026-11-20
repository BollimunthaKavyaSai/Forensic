"""
Revision R3 experiments: leakage-controlled, repeated stratified CV, tabular baselines,
graph models, overfitting diagnostics, and significance testing.

Usage:
  python run_experiments.py windows  <path to cybercrime_forensic_dataset.csv>
  python run_experiments.py android  <path to ADROIT dataset.csv>
Outputs per-fold CSVs in ./results/.
"""
import sys, os, time, json, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score,
                             matthews_corrcoef, precision_score, recall_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import kneighbors_graph
from xgboost import XGBClassifier
from data_prep import load_windows, load_adroit

warnings.filterwarnings("ignore")
N_SPLITS, N_REPEATS = 5, 3
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)


def metrics(y, proba):
    pred = proba.argmax(1)
    m = {"acc": accuracy_score(y, pred), "bal_acc": balanced_accuracy_score(y, pred),
         "macro_f1": f1_score(y, pred, average="macro"), "weighted_f1": f1_score(y, pred, average="weighted"),
         "mcc": matthews_corrcoef(y, pred)}
    if proba.shape[1] == 2:
        m["auc"] = roc_auc_score(y, proba[:, 1])
        m["precision"] = precision_score(y, pred, zero_division=0)
        m["recall"] = recall_score(y, pred, zero_division=0)
        m["f1"] = f1_score(y, pred, zero_division=0)
    else:
        try:
            m["auc"] = roc_auc_score(y, proba, multi_class="ovr", average="macro")
        except ValueError:
            m["auc"] = np.nan
    return m


# ---------------------------------------------------------------- sklearn models
def sk_models(seed, n_classes):
    return {
        "Majority": DummyClassifier(strategy="most_frequent"),
        "LogReg": LogisticRegression(max_iter=2000, C=1.0),
        "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=4, random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9,
                                 colsample_bytree=0.9, n_jobs=4, random_state=seed, verbosity=0,
                                 eval_metric="logloss"),
    }


# ---------------------------------------------------------------- graph models
from torch_geometric.nn import GCNConv, GATConv


class GCN(nn.Module):
    def __init__(s, i, c, h=64, p=0.5):
        super().__init__(); s.c1 = GCNConv(i, h); s.c2 = GCNConv(h, c); s.p = p
    def forward(s, x, ei):
        return s.c2(F.dropout(F.relu(s.c1(x, ei)), s.p, s.training), ei)


class GAT(nn.Module):
    def __init__(s, i, c, h=16, heads=8, p=0.5):
        super().__init__()
        s.g1 = GATConv(i, h, heads=heads, dropout=p); s.g2 = GATConv(h * heads, c, heads=1, concat=False, dropout=p); s.p = p
    def forward(s, x, ei):
        x = F.dropout(x, s.p, s.training)
        return s.g2(F.dropout(F.elu(s.g1(x, ei)), s.p, s.training), ei)


class GraphMLP(nn.Module):  # same width as GCN, no graph: isolates the contribution of graph construction
    def __init__(s, i, c, h=64, p=0.5):
        super().__init__(); s.l1 = nn.Linear(i, h); s.l2 = nn.Linear(h, c); s.p = p
    def forward(s, x, ei=None):
        return s.l2(F.dropout(F.relu(s.l1(x)), s.p, s.training))


def train_graph(model_cls, X, y, tr, va, te, n_classes, seed, k=8, max_ep=300, patience=20,
                lr=5e-3, wd=5e-4, use_graph=True):
    set_seed(seed)
    sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X).astype(np.float32)
    A = kneighbors_graph(Xs, n_neighbors=k, mode="connectivity", include_self=False)
    A = A.maximum(A.T)  # symmetrise
    ei = torch.tensor(np.vstack(A.nonzero()), dtype=torch.long)
    x = torch.tensor(Xs); yt = torch.tensor(y, dtype=torch.long)
    model = model_cls(X.shape[1], n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best, best_state, bad, best_ep = 1e9, None, 0, 0
    hist = []
    for ep in range(max_ep):
        model.train(); opt.zero_grad()
        out = model(x, ei); loss = F.cross_entropy(out[tr], yt[tr]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out = model(x, ei); vl = F.cross_entropy(out[va], yt[va]).item()
            pr = out.argmax(1)
            hist.append((ep, loss.item(), vl, (pr[tr] == yt[tr]).float().mean().item(),
                         (pr[va] == yt[va]).float().mean().item()))
        if vl < best - 1e-4:
            best, best_state, bad, best_ep = vl, {k_: v.clone() for k_, v in model.state_dict().items()}, 0, ep
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        P = F.softmax(model(x, ei), 1).numpy()
    return P, {"best_epoch": best_ep, "epochs_run": ep + 1,
               "train_acc": accuracy_score(y[tr], P[tr].argmax(1)),
               "val_acc": accuracy_score(y[va], P[va].argmax(1))}, hist


# ---------------------------------------------------------------- tabular transformers
class TabTransformerOriginal(nn.Module):
    """Exact architecture of ai_ml/android_model_infer.py in the repository (single token)."""
    def __init__(s, d_in, n_classes=2, e=128, heads=8, layers=2, p=0.3):
        super().__init__()
        s.emb = nn.Linear(d_in, e)
        s.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(e, heads, dropout=p, batch_first=True), layers)
        s.do = nn.Dropout(p); s.fc = nn.Linear(e, n_classes)
    def forward(s, x):
        return s.fc(s.do(s.tr(s.emb(x).unsqueeze(1)).squeeze(1)))


class FTTransformer(nn.Module):
    """Feature-tokenised Transformer (Gorishniy et al., 2021): one token per feature + [CLS]."""
    def __init__(s, d_in, n_classes=2, d=32, heads=4, layers=2, p=0.1):
        super().__init__()
        s.W = nn.Parameter(torch.randn(d_in, d) * 0.02); s.b = nn.Parameter(torch.zeros(d_in, d))
        s.cls = nn.Parameter(torch.zeros(1, 1, d))
        s.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=p,
                                                                batch_first=True, norm_first=True), layers)
        s.norm = nn.LayerNorm(d); s.fc = nn.Linear(d, n_classes)
    def forward(s, x):
        t = x.unsqueeze(-1) * s.W + s.b
        t = torch.cat([s.cls.expand(x.shape[0], -1, -1), t], 1)
        return s.fc(F.relu(s.norm(s.tr(t)[:, 0])))


class TabMLP(nn.Module):
    def __init__(s, d_in, n_classes=2, h=128, p=0.3):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, h), nn.ReLU(),
                              nn.Dropout(p), nn.Linear(h, n_classes))
    def forward(s, x): return s.net(x)


def train_tab(model_cls, Xtr, ytr, Xva, yva, Xte, n_classes, seed, max_ep=60, patience=8, lr=1e-3, wd=1e-4, bs=512):
    set_seed(seed)
    model = model_cls(Xtr.shape[1], n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr)
    xv, yv = torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva)
    best, best_state, bad, best_ep = 1e9, None, 0, 0
    for ep in range(max_ep):
        model.train(); perm = torch.randperm(len(xt))
        for i in range(0, len(xt), bs):
            idx = perm[i:i + bs]; opt.zero_grad()
            F.cross_entropy(model(xt[idx]), yt[idx]).backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = F.cross_entropy(model(xv), yv).item()
        if vl < best - 1e-4:
            best, best_state, bad, best_ep = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0, ep
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        f = lambda X: F.softmax(model(torch.tensor(X, dtype=torch.float32)), 1).numpy()
        return f(Xte), {"best_epoch": best_ep, "epochs_run": ep + 1,
                        "train_acc": accuracy_score(ytr, f(Xtr).argmax(1)),
                        "val_acc": accuracy_score(yva, f(Xva).argmax(1))}


# ---------------------------------------------------------------- Windows track
def run_windows(path):
    torch.set_num_threads(5)
    rows, hists = [], {}
    configs = [("clean", False)] + [("leaky", True)]
    for feat_set, leaky in configs:
        X, yb, ym = load_windows(path, with_leaky=leaky)
        X = X.values
        for task, y in [("binary", yb), ("multiclass", ym)]:
            if feat_set == "leaky" and task == "binary":
                continue  # leakage ablation reported for the original 5-class task only
            C = len(np.unique(y))
            rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=0)
            for f, (trv, te) in enumerate(rskf.split(X, y)):
                seed = f
                tr, va = train_test_split(trv, test_size=0.15, stratify=y[trv], random_state=seed)
                names = ["Majority", "LogReg", "RandomForest", "XGBoost"] if feat_set == "clean" else ["RandomForest"]
                sc = StandardScaler().fit(X[trv])
                for nm in names:
                    m = sk_models(seed, C)[nm]; t0 = time.time()
                    m.fit(sc.transform(X[trv]), y[trv]); P = m.predict_proba(sc.transform(X[te]))
                    tr_acc = accuracy_score(y[trv], m.predict(sc.transform(X[trv])))
                    rows.append(dict(track="windows", feat=feat_set, task=task, fold=f, model=nm,
                                     train_acc=tr_acc, time_s=time.time() - t0, **metrics(y[te], P)))
                gnames = [("MLP(no graph)", GraphMLP, False), ("GCN", GCN, True), ("GAT", GAT, True)]
                if feat_set == "leaky": gnames = [("GAT", GAT, True)]
                for nm, cls, g in gnames:
                    t0 = time.time()
                    P, info, hist = train_graph(cls, X, y, tr, va, te, C, seed)
                    rows.append(dict(track="windows", feat=feat_set, task=task, fold=f, model=nm,
                                     time_s=time.time() - t0, **info, **metrics(y[te], P[te])))
                    if f == 0: hists[f"{feat_set}_{task}_{nm}"] = hist
                print(feat_set, task, f, "done", flush=True)
                pd.DataFrame(rows).to_csv(os.path.join(OUT, "windows_folds.csv"), index=False)
    json.dump(hists, open(os.path.join(OUT, "windows_curves_fold0.json"), "w"))
    run_windows_extra(path)


def run_windows_extra(path):
    torch.set_num_threads(4)
    # Resample-before-split ablation: reproduces the protocol of the original notebook
    X, yb, ym = load_windows(path, with_leaky=True); X = X.values
    rng = np.random.RandomState(0); idx = []
    for c in np.unique(ym):
        ic = np.where(ym == c)[0]; idx.append(rng.choice(ic, 6167, replace=len(ic) < 6167))
    idx = np.concatenate(idx); Xr, yr = X[idx], ym[idx]
    rr = []
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=1, random_state=0)
    for f, (trv, te) in enumerate(rskf.split(Xr, yr)):
        tr, va = train_test_split(trv, test_size=0.15, stratify=yr[trv], random_state=f)
        P, info, _ = train_graph(GAT, Xr, yr, tr, va, te, 5, f)
        rr.append(dict(model="GAT", protocol="oversample-before-split+leaky", fold=f, **metrics(yr[te], P[te])))
        m = RandomForestClassifier(300, n_jobs=4, random_state=f).fit(Xr[trv], yr[trv])
        rr.append(dict(model="RandomForest", protocol="oversample-before-split+leaky", fold=f,
                       **metrics(yr[te], m.predict_proba(Xr[te]))))
    pd.DataFrame(rr).to_csv(os.path.join(OUT, "windows_resample_ablation.csv"), index=False)

    # Label-permutation test of feature-label dependence (clean features, binary task)
    X, yb, _ = load_windows(path); X = X.values
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    rf = RandomForestClassifier(200, n_jobs=4, random_state=0)
    obs = cross_val_score(rf, X, yb, cv=cv, scoring="roc_auc").mean()
    null = [cross_val_score(rf, X, np.random.RandomState(i).permutation(yb), cv=cv, scoring="roc_auc").mean()
            for i in range(100)]
    p = (1 + sum(n >= obs for n in null)) / 101
    from sklearn.feature_selection import mutual_info_classif, chi2
    mi = mutual_info_classif(X, yb, random_state=0)
    json.dump({"observed_auc": obs, "null_mean": float(np.mean(null)), "null_std": float(np.std(null)),
               "p_value": p, "null_distribution": [float(v) for v in null], "max_mutual_info": float(mi.max()), "mean_mutual_info": float(mi.mean())},
              open(os.path.join(OUT, "windows_permutation_test.json"), "w"), indent=1)


# ---------------------------------------------------------------- Android track
STORE_META = ["rating_number", "rating_count", ".//Min_OpenGL", ".//State"]


def rf_select(X, y, k, seed):
    rf = RandomForestClassifier(n_estimators=300, n_jobs=6, random_state=seed).fit(X, y)
    return np.argsort(-rf.feature_importances_)[:k]


def run_android(path, job=None):
    torch.set_num_threads(3)
    num, y, groups = load_adroit(path)
    cols_all = list(num.columns)
    sets = {"all169": cols_all, "device_derivable": [c for c in cols_all if c not in STORE_META]}
    rows = []
    protocols = [("random", RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=0)),
                 ("dedup_grouped", None)]
    for proto, splitter in protocols:
        if proto == "random":
            splits = list(splitter.split(num, y))
        else:
            splits = []
            for r in range(N_REPEATS):
                sgk = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=r)
                splits += list(sgk.split(num, y, groups))
        for fs_name, cols in sets.items():
            if fs_name == "device_derivable" and proto == "random":
                continue
            if job and job != f"{proto}_{fs_name}":
                continue
            Xall = num[cols].values.astype(float)
            out_csv = os.path.join(OUT, f"android_folds_{job or 'all'}.csv")
            done = set()
            if os.path.exists(out_csv):  # resume: keep folds whose last model (FT-Transformer) finished
                prev = pd.read_csv(out_csv)
                prev = prev[(prev.protocol == proto) & (prev.feat == fs_name)]
                done = set(prev[prev.model == "FT-Transformer"].fold)
                rows += prev[prev.fold.isin(done)].to_dict("records")
                print("resuming; completed folds:", sorted(done), flush=True)
            for f, (trv, te) in enumerate(splits):
                seed = f
                if f in done:
                    continue
                med = np.nanmedian(Xall[trv], axis=0); med = np.where(np.isnan(med), 0, med)
                Xi = np.where(np.isnan(Xall), med, Xall)
                sel = rf_select(Xi[trv], y[trv], 51, seed)  # selection inside the training fold only
                Xs = Xi[:, sel]
                sc = StandardScaler().fit(Xs[trv]); Xz = sc.transform(Xs).astype(np.float32)
                tr, va = train_test_split(trv, test_size=0.15, stratify=y[trv], random_state=seed)
                models = dict(sk_models(seed, 2)); models["SVM-RBF"] = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed)
                for nm, m in models.items():
                    t0 = time.time(); m.fit(Xz[trv], y[trv])
                    if hasattr(m, "predict_proba"):
                        P = m.predict_proba(Xz[te])
                    else:  # SVM: logistic squashing of the decision function (argmax unchanged)
                        s_ = 1 / (1 + np.exp(-m.decision_function(Xz[te]))); P = np.c_[1 - s_, s_]
                    rows.append(dict(track="android", protocol=proto, feat=fs_name, fold=f, model=nm,
                                     train_acc=accuracy_score(y[trv], m.predict(Xz[trv])),
                                     time_s=time.time() - t0, **metrics(y[te], P)))
                    print("  ", nm, round(time.time() - t0, 1), flush=True)
                for nm, cls in [("MLP", TabMLP), ("TabTransformer(single-token, original)", TabTransformerOriginal),
                                ("FT-Transformer", FTTransformer)]:
                    t0 = time.time()
                    P, info = train_tab(cls, Xz[tr], y[tr], Xz[va], y[va], Xz[te], 2, seed)
                    rows.append(dict(track="android", protocol=proto, feat=fs_name, fold=f, model=nm,
                                     time_s=time.time() - t0, **info, **metrics(y[te], P)))
                    print("  ", nm, round(time.time() - t0, 1), info["epochs_run"], flush=True)
                print(proto, fs_name, f, "done", flush=True)
                pd.DataFrame(rows).to_csv(os.path.join(OUT, f"android_folds_{job or 'all'}.csv"), index=False)
    if job and job != "random_all169":
        return
    # duplicate statistics
    te_dup = []
    for f, (trv, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=0).split(num, y)):
        te_dup.append(np.isin(groups[te], groups[trv]).mean())
    json.dump({"n_rows": int(len(y)), "n_unique_feature_vectors": int(groups.max() + 1),
               "frac_test_rows_with_identical_train_row_random_split": float(np.mean(te_dup)),
               "n_label_conflicting_groups": int(pd.Series(y).groupby(groups).nunique().gt(1).sum())},
              open(os.path.join(OUT, "android_duplicate_audit.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "windows":
        run_windows(sys.argv[2])
    elif sys.argv[1] == "windows_extra":
        run_windows_extra(sys.argv[2])
    else:
        run_android(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
