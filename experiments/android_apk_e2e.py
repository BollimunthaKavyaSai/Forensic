"""
Revision R4: end-to-end Android path on real APK files.
1. Restrict KronoDroid to the 'deployable static subset' that the prototype can compute from an APK
   (prototype_ext/apk_features.py) and evaluate it with family-aware grouped CV (unseen malware families).
2. Train the final model on all KronoDroid apps and run it, through the prototype's APK feature extractor, on
   benign open-source APKs downloaded from F-Droid (known-benign ground truth) to measure the false-positive rate
   on real, recent APK files.
Usage: python android_apk_e2e.py
"""
import os, glob, json, time
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from run_experiments import metrics
from run_kronodroid import load
from prototype_ext.apk_features import deployable_columns, apk_features

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "results")
a, feats, static, dyn, y, year, groups = load()
cols = deployable_columns(static)
X = a[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(float)
mk = lambda s: XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                             n_jobs=6, random_state=s, verbosity=0, eval_metric="logloss")
rows = []
for r in range(3):
    for i, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=r).split(X, y, groups)):
        for nm, m in [("XGBoost", mk(r * 5 + i)), ("RandomForest", RandomForestClassifier(300, n_jobs=6, random_state=r * 5 + i))]:
            m.fit(X[tr], y[tr]); P = m.predict_proba(X[te])
            rows.append(dict(model=nm, fold=r * 5 + i, **metrics(y[te], P)))
cv = pd.DataFrame(rows); cv.to_csv(os.path.join(OUT, "android_deployable_family_cv.csv"), index=False)
print(cv.groupby("model")[["acc", "f1", "auc"]].agg(["mean", "std"]).round(4))

final = mk(0).fit(X, y)
# threshold at which the family-CV false-positive rate on benign apps is ~1% would need CV scores; use 0.5 (default)
apks = sorted(glob.glob(os.path.join(HERE, "..", "data_real", "fdroid", "apks", "*.apk")))
res = []
for p in apks:
    t0 = time.time()
    try:
        f, meta = apk_features(p, cols)
        s = float(final.predict_proba(np.array([[f[c] for c in cols]], dtype=float))[0, 1])
        res.append(dict(apk=os.path.basename(p), score=s, flagged=int(s >= 0.5), seconds=time.time() - t0, **meta,
                        n_perm_flags=int(sum(f[c] for c in cols if c.isupper()))))
    except Exception as e:
        res.append(dict(apk=os.path.basename(p), error=repr(e)))
R = pd.DataFrame(res); R.to_csv(os.path.join(OUT, "android_fdroid_e2e.csv"), index=False)
ok = R[R.get("score").notna()] if "score" in R else R.iloc[0:0]
summ = dict(n_features=len(cols), n_apks=len(apks), n_parsed=int(len(ok)),
            false_positive_rate=float(ok.flagged.mean()) if len(ok) else None,
            median_score=float(ok.score.median()) if len(ok) else None,
            mean_seconds_per_apk=float(ok.seconds.mean()) if len(ok) else None,
            family_cv=cv.groupby("model")[["acc", "f1", "auc"]].mean().round(4).to_dict(orient="index"))
json.dump(summ, open(os.path.join(OUT, "android_fdroid_e2e.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
