"""
Revision R4: Android track on KronoDroid real-device data (Guerra-Manzanares et al., Comput. Secur. 2021).
Features were recorded from APKs executed on real Android handsets (system calls) and from the APK itself
(manifest permissions, components, file sizes). Columns derived from antivirus verdicts or identifiers are removed.

Protocols
  random   : stratified 5-fold x 3
  family   : stratified group 5-fold x 3, malware grouped by family, so test families are unseen in training
  temporal : train on apps last modified in years <= Y, test on year Y+1 (Y = 2012..2019), TESSERACT-style
Feature sets
  all      : dynamic (system calls, contacted IPs) + static (APK-derivable)
  static   : APK-derivable only, i.e. what can be computed from an APK recovered from a seized device
Usage: python run_kronodroid.py <protocol> <featureset>
"""
import os, sys, time, json
import numpy as np, pandas as pd, torch
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from run_experiments import (metrics, sk_models, rf_select, train_tab, TabMLP, TabTransformerOriginal,
                             FTTransformer, OUT)

HERE = os.path.dirname(os.path.abspath(__file__))
KD = os.path.join(HERE, "..", "data_real", "kronodroid")
LEAKY_OR_ID = ["Package", "sha256", "EarliestModDate", "HighestModDate", "TimesSubmitted", "Scanners",
               "Detection_Ratio", "MalFamily", "Malware"]


def load():
    a = pd.concat([pd.read_csv(os.path.join(KD, "real_legitimate_v1.csv"), low_memory=False),
                   pd.read_csv(os.path.join(KD, "real_malware_v1.csv"), low_memory=False)], ignore_index=True)
    a = a.drop_duplicates(subset="sha256").reset_index(drop=True)
    cols = list(a.columns)
    dyn = cols[cols.index("execve"):cols.index("nr_syscalls") + 1] + ["NrContactedIps"]
    feats = [c for c in cols if c not in LEAKY_OR_ID]
    static = [c for c in feats if c not in dyn]
    year = pd.to_datetime(a.HighestModDate, errors="coerce", format="%m/%d/%Y").dt.year
    fam = a.MalFamily.fillna("").astype(str)
    grp = np.where((a.Malware == 1) & (fam != ""), "F:" + fam, "S:" + a.sha256.astype(str))
    return a, feats, static, dyn, a.Malware.values.astype(int), year.values, pd.factorize(grp)[0]


def splits_for(proto, X, y, year, groups):
    if proto == "random":
        return [(f"r{i}", tr, te) for i, (tr, te) in
                enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0).split(X, y))]
    if proto == "family":
        out = []
        for r in range(3):
            for i, (tr, te) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=r).split(X, y, groups)):
                out.append((f"g{r * 5 + i}", tr, te))
        return out
    if proto == "temporal":
        ok = (year >= 2008) & (year <= 2020)
        out = []
        for Y in range(2012, 2020):
            tr = np.where(ok & (year <= Y))[0]; te = np.where(ok & (year == Y + 1))[0]
            if len(np.unique(y[te])) == 2:
                out.append((f"t{Y + 1}", tr, te))
        return out
    raise ValueError(proto)


def main(proto, fs):
    torch.set_num_threads(int(os.environ.get("NTHREADS", "3")))
    a, feats, static, dyn, y, year, groups = load()
    cols = feats if fs == "all" else static
    X = a[cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
    out_csv = os.path.join(OUT, f"kronodroid_folds_{proto}_{fs}{'_ft' if os.environ.get('FT_ONLY') == '1' else ''}.csv")
    rows, done = [], set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv); done = set(prev[prev.model == "FOLD_DONE"].fold)
        rows = prev[prev.fold.isin(done)].to_dict("records"); print("resuming", sorted(done), flush=True)
    for k, (fold, trv, te) in enumerate(splits_for(proto, X, y, year, groups)):
        if k >= int(os.environ.get("MAX_FOLDS", "99")): break
        if fold in done: continue
        seed = k
        med = np.nanmedian(X[trv], axis=0); med = np.where(np.isnan(med), 0, med)
        Xi = np.where(np.isnan(X), med, X)
        sel = rf_select(Xi[trv], y[trv], 51, seed)
        Xs = Xi[:, sel]; sc = StandardScaler().fit(Xs[trv]); Xz = sc.transform(Xs).astype(np.float32)
        tr, va = train_test_split(trv, test_size=0.15, stratify=y[trv], random_state=seed)
        base = dict(track="kronodroid", protocol=proto, feat=fs, fold=fold, n_train=len(trv), n_test=len(te),
                    test_malware_frac=float(y[te].mean()))
        for nm, m in (sk_models(seed, 2).items() if os.environ.get("FT_ONLY") != "1" else []):
            t0 = time.time(); m.fit(Xz[trv], y[trv]); P = m.predict_proba(Xz[te])
            rows.append(dict(**base, model=nm, train_acc=accuracy_score(y[trv], m.predict(Xz[trv])),
                             time_s=time.time() - t0, **metrics(y[te], P)))
            print("  ", fold, nm, round(time.time() - t0, 1), flush=True)
        nets = [("MLP", TabMLP)] if os.environ.get("FT_ONLY") != "1" else []
        if os.environ.get("FT_ONLY") == "1":
            nets = [("FT-Transformer", FTTransformer)]
        for nm, cls in nets:
            t0 = time.time()
            P, info = train_tab(cls, Xz[tr], y[tr], Xz[va], y[va], Xz[te], 2, seed, max_ep=30, patience=5, bs=1024)
            rows.append(dict(**base, model=nm, time_s=time.time() - t0, **info, **metrics(y[te], P)))
            print("  ", fold, nm, round(time.time() - t0, 1), info["epochs_run"], flush=True)
        rows.append(dict(**base, model="FOLD_DONE"))
        pd.DataFrame(rows).to_csv(out_csv, index=False); print(proto, fs, fold, "done", flush=True)
    if proto == "random" and fs == "all":
        json.dump({"n_apps": int(len(y)), "n_malware": int(y.sum()), "n_benign": int((1 - y).sum()),
                   "n_features_all": len(feats), "n_features_static": len(static), "n_features_dynamic": len(dyn),
                   "n_families": int(a.MalFamily.nunique()),
                   "year_counts": pd.crosstab(year, y).to_dict()},
                  open(os.path.join(OUT, "kronodroid_profile.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
