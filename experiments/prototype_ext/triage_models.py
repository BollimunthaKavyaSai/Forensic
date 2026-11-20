"""
Revision R4: model side of the end-to-end prototype.

train_and_save()  - trains the deployed models on all labelled data and writes them to prototype_ext/models/
    * Windows: process-level classifier on the 51 behavioural features of atlas_parse.py (all 16 ATLASv2 scenarios)
    * Android: classifier on the deployable static APK features (all KronoDroid apps)
windows_triage(xml_or_evtx_records) - security/sysmon events -> process provenance graph -> ranked processes
android_triage(apk_paths)            - APK files -> static features -> malware likelihood per APK

The functions return dictionaries shaped like the prototype's original ai_ml inference results
(success, total_activities / total_samples, anomalies_detected, top_anomalies) so the Django endpoints
/api/cases/<id>/ml-infer/ and /android-ml-infer/ can call them unchanged.
"""
import os, sys, pickle, tempfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)
MODELS = os.path.join(HERE, "models"); os.makedirs(MODELS, exist_ok=True)


def train_and_save(win_model="RandomForest", exclude_vector=None):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from xgboost import XGBClassifier
    # Windows
    G = pickle.load(open(os.path.join(EXP, "results", "atlas_graphs.pkl"), "rb"))
    feat = [c for c in next(iter(G.values()))["proc"].columns if c not in ("pid", "image", "scen", "kind", "label")]
    FLASH = {"s1", "s2", "m1", "m2", "m3"}
    keep = [s for s in G if exclude_vector is None or (("flash" if s.split("-")[1] in FLASH else "office") != exclude_vector)]
    P = pd.concat([G[s]["proc"] for s in keep]); P = P[P.label.isin(["attack", "benign"])]
    X = np.log1p(P[feat].values.astype(float).clip(min=0)); y = (P.label == "attack").astype(int).values
    m = {"RandomForest": RandomForestClassifier(500, class_weight="balanced_subsample", n_jobs=4, random_state=0),
         "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))}[win_model]
    m.fit(X, y); pickle.dump(dict(model=m, features=feat, name=win_model), open(os.path.join(MODELS, "windows_triage.pkl" if exclude_vector is None else f"windows_triage_no_{exclude_vector}.pkl"), "wb"))
    if exclude_vector is not None: return
    # Android
    from run_kronodroid import load
    from prototype_ext.apk_features import deployable_columns
    a, feats, static, dyn, ya, year, groups = load()
    cols = deployable_columns(static)
    Xa = a[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(float)
    xa = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                       n_jobs=4, random_state=0, verbosity=0, eval_metric="logloss").fit(Xa, ya)
    pickle.dump(dict(model=xa, features=cols, name="XGBoost"), open(os.path.join(MODELS, "android_triage.pkl"), "wb"))
    return os.path.join(MODELS, "windows_triage.pkl"), os.path.join(MODELS, "android_triage.pkl")


def windows_triage(security_xml, threshold=0.5, top_n=50):
    """security_xml: path to an exported Security log (XML); a sysmon export with the same name pattern is used if present."""
    import atlas_parse
    W = pickle.load(open(os.path.join(MODELS, os.environ.get("WIN_MODEL_FILE", "windows_triage.pkl")), "rb"))
    df, edges, files, ends = atlas_parse.parse_scenario(security_xml)
    X = np.log1p(df[W["features"]].values.astype(float).clip(min=0))
    s = W["model"].predict_proba(X)[:, 1]
    df = df.assign(score=s).sort_values("score", ascending=False)
    top = [dict(process_id=int(r.pid), image=r.image, score=round(float(r.score), 4), n_events=int(r.n_events),
                n_children=int(r.n_children), n_remote_ips=int(r.n_remote_ips), n_files=int(r.n_files))
           for r in df.head(top_n).itertuples()]
    return dict(success=True, model=W["name"], total_activities=int(len(df)), anomalies_detected=int((s >= threshold).sum()),
                threshold=threshold, top_anomalies=top, graph_edges=len(edges))


def android_triage(apk_paths, threshold=0.5, top_n=50):
    from prototype_ext.apk_features import apk_features
    A = pickle.load(open(os.path.join(MODELS, "android_triage.pkl"), "rb"))
    rows = []
    for p in apk_paths:
        try:
            f, meta = apk_features(p, A["features"])
            s = float(A["model"].predict_proba(np.array([[f[c] for c in A["features"]]], dtype=float))[0, 1])
            rows.append(dict(apk=os.path.basename(p), package=meta["package"], score=round(s, 4)))
        except Exception as e:
            rows.append(dict(apk=os.path.basename(p), error=repr(e)))
    ok = sorted([r for r in rows if "score" in r], key=lambda r: -r["score"])
    return dict(success=True, model=A["name"], total_samples=len(rows), anomalies_detected=sum(r["score"] >= threshold for r in ok),
                threshold=threshold, top_anomalies=ok[:top_n], errors=[r for r in rows if "error" in r])


if __name__ == "__main__":
    print(train_and_save(sys.argv[1] if len(sys.argv) > 1 else "RandomForest", sys.argv[2] if len(sys.argv) > 2 else None))
