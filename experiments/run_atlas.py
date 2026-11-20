"""
Revision R4: Windows track on real host audit logs (ATLASv2). Process nodes are classified as attack vs. benign
(REAPr labels; 'contaminated' and unlabelled processes are excluded from loss and metrics).

Graph models use the provenance graph recovered from the logs themselves (parent->child, process-file,
process-endpoint), trained inductively on whole scenario graphs. Baselines: tabular models on the identical process
features, and the k-NN feature-similarity graph used in the earlier version of the paper.

Protocols (one test scenario per fold, 16 folds):
  cross_vector : train only on scenarios whose initial exploit vector differs from the test scenario (Flash vs Office)
  loso         : train on all other scenarios (attack chains are near-duplicates; reported for comparison only)
Usage: python run_atlas.py <protocol>
"""
import os, sys, time, pickle, json
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import kneighbors_graph
from xgboost import XGBClassifier
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "results")
SEEDS = [int(x) for x in os.environ.get('SEEDS', '0,1,2,3,4').split(',')]
OFFICE = ("winword.exe", "excel.exe", "eqnedt32.exe", "powerpnt.exe")
HUB = 30  # file/endpoint nodes touched by more than HUB processes in a scenario are dropped (system DLLs etc.)


def load():
    G = pickle.load(open(os.path.join(OUT, "atlas_graphs.pkl"), "rb"))
    att = {s: g for s, g in G.items() if g["kind"] == "attack"}
    ben = {s: g for s, g in G.items() if g["kind"] == "benign"}
    feat = [c for c in next(iter(G.values()))["proc"].columns if c not in ("pid", "image", "scen", "kind", "label")]
    if os.environ.get("FEATSET") == "nopath":  # ablation: drop executable-location features (img_*)
        feat = [c for c in feat if not c.startswith("img_")]
    # initial exploit vector of each attack, from the ATLAS attack descriptions (Metasploit module used in step 2):
    # S1, S2, M1-M3 = Firefox/Adobe Flash exploits; S3, S4, M4-M6 = Microsoft Office exploits (CVE-2017-11882,
    # office_word_hta, CVE-2018-8174). Host 2 (lateral-movement victim) inherits the vector of its scenario.
    FLASH = {"s1", "s2", "m1", "m2", "m3"}
    vec = {s: ("flash" if s.split("-")[1] in FLASH else "office") for s in att}
    return att, ben, feat, vec


def build(g, feat):
    """Return node features (process rows first), edge_index, process labels (1/0/-1)."""
    P = g["proc"].reset_index(drop=True); n = len(P)
    key = {(p, i): j for j, (p, i) in enumerate(zip(P.pid, P.image))}
    X = np.log1p(P[feat].values.astype(float).clip(min=0))
    ent, edges = {}, []
    for (a, b) in g["edges"]:
        if a in key and b in key: edges.append((key[a], key[b]))
    cnt = {}
    for d in (g["files"], g["ends"]):
        for k, objs in d.items():
            for o in objs: cnt[o] = cnt.get(o, 0) + 1
    for typ, d in (("f", g["files"]), ("e", g["ends"])):
        for k, objs in d.items():
            if k not in key: continue
            for o in objs:
                if cnt[o] > HUB or cnt[o] < 2: continue  # keep only entities shared by 2..HUB processes
                if (typ, o) not in ent: ent[(typ, o)] = n + len(ent)
                edges.append((key[k], ent[(typ, o)]))
    m = len(ent); Xe = np.zeros((m, X.shape[1]))
    T = np.zeros((n + m, 3)); T[:n, 0] = 1
    for (typ, o), j in ent.items(): T[j, 1 if typ == "f" else 2] = 1
    Xall = np.hstack([np.vstack([X, Xe]), T]).astype(np.float32)
    ei = np.array(edges).T if edges else np.zeros((2, 0), int)
    ei = np.hstack([ei, ei[::-1]]) if ei.size else ei
    y = P.label.map({"attack": 1, "benign": 0}).fillna(-1).values.astype(int)
    return Xall, ei, y, n


class GNN(nn.Module):
    def __init__(s, kind, d, h=64, p=0.3):
        super().__init__(); s.kind = kind
        if kind == "GAT": s.c1, s.c2 = GATConv(d, 16, heads=4, dropout=p), GATConv(64, h, heads=1, dropout=p)
        elif kind == "SAGE": s.c1, s.c2 = SAGEConv(d, h), SAGEConv(h, h)
        else: s.c1, s.c2 = GCNConv(d, h), GCNConv(h, h)
        s.skip = nn.Linear(d, h); s.out = nn.Linear(h, 2); s.p = p
    def forward(s, x, ei):
        h = F.elu(s.c1(F.dropout(x, s.p, s.training), ei)); h = F.dropout(h, s.p, s.training)
        h = F.elu(s.c2(h, ei) + s.skip(x)); return s.out(h)


class MLP(nn.Module):
    def __init__(s, d, h=64, p=0.3):
        super().__init__(); s.net = nn.Sequential(nn.Linear(d, h), nn.ELU(), nn.Dropout(p), nn.Linear(h, h), nn.ELU(),
                                                  nn.Dropout(p), nn.Linear(h, 2))
    def forward(s, x, ei=None): return s.net(x)


def knn_ei(X, k=8):
    A = kneighbors_graph(X, min(k, len(X) - 1), include_self=False).tocoo()
    ei = np.vstack([A.row, A.col]); return np.hstack([ei, ei[::-1]])


def train_nn(make, graphs_tr, graphs_va, graphs_te, seed, max_ep=150, patience=15, lr=5e-3, wd=5e-4):
    torch.manual_seed(seed); np.random.seed(seed)
    model = make(); opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    ytr = np.concatenate([g[2][g[2] >= 0] for g in graphs_tr]); w = len(ytr) / (2 * np.bincount(ytr, minlength=2) + 1e-9)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    T = lambda g: (torch.tensor(g[0]), torch.tensor(g[1], dtype=torch.long), torch.tensor(g[2]))
    tr, va, te = [T(g) for g in graphs_tr], [T(g) for g in graphs_va], [T(g) for g in graphs_te]
    best, bad, state, ep_best = 1e9, 0, None, 0
    for ep in range(max_ep):
        model.train(); opt.zero_grad(); loss = 0
        for x, ei, y in tr:
            o = model(x, ei)[: len(y)]; mk = y >= 0; loss = loss + crit(o[mk], y[mk])
        (loss / len(tr)).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = sum(crit(model(x, ei)[: len(y)][y >= 0], y[y >= 0]).item() for x, ei, y in va) / len(va)
        if vl < best - 1e-4: best, bad, ep_best = vl, 0, ep; state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        sv = [F.softmax(model(x, ei)[: len(y)], 1)[:, 1].numpy() for x, ei, y in va]
        st = [F.softmax(model(x, ei)[: len(y)], 1)[:, 1].numpy() for x, ei, y in te]
    return sv, st, dict(best_epoch=ep_best, epochs_run=ep + 1)


def thr_from_val(yv, sv):
    best, t = -1, 0.5
    for c in np.unique(np.round(sv, 4)):
        f = f1_score(yv, sv >= c, zero_division=0)
        if f > best: best, t = f, c
    return t


def score(y, s, t):
    mk = y >= 0; y, s = y[mk], s[mk]; order = np.argsort(-s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    k5 = max(1, int(np.ceil(0.05 * len(s))))
    return dict(auc=roc_auc_score(y, s), ap=average_precision_score(y, s), f1=f1_score(y, s >= t, zero_division=0),
                precision=precision_score(y, s >= t, zero_division=0), recall=recall_score(y, s >= t, zero_division=0),
                recall_at_5pct=float(y[order[:k5]].sum() / y.sum()),
                effort_full_recall=float(ranks[y == 1].max() / len(s)), n_test=int(len(y)), n_attack=int(y.sum()),
                base_rate=float(y.mean()))


def main(proto):
    torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
    att, ben, feat, vec = load()
    scen = sorted(att)
    built = {s: build(att[s], feat) for s in scen}
    rows = []; out_csv = os.path.join(OUT, f"atlas_folds_{proto}{'_nopath' if os.environ.get('FEATSET') == 'nopath' else ''}.csv")
    for s_te in scen:
        pool = [s for s in scen if s != s_te and (proto == "loso" or vec[s] != vec[s_te])]
        for seed in SEEDS:
            rng = np.random.RandomState(seed); s_va = [pool[rng.randint(len(pool))]]; s_tr = [s for s in pool if s not in s_va]
            # standardise process features with training statistics (entity rows stay zero; type flags untouched)
            Xp = np.vstack([built[s][0][: built[s][3]] for s in s_tr]); sc = StandardScaler().fit(Xp[:, :-3])
            def z(s):
                X, ei, y, n = built[s]; X = X.copy(); X[:n, :-3] = sc.transform(X[:n, :-3]); return X, ei, y, n
            Gtr, Gva, Gte = [z(s) for s in s_tr], [z(s) for s in s_va], [z(s_te)]
            proc = lambda G: (np.vstack([g[0][: g[3]] for g in G]), np.concatenate([g[2] for g in G]))
            Xtr, ytr = proc(Gtr); Xva, yva = proc(Gva); Xte, yte = proc(Gte)
            mtr = ytr >= 0
            base = dict(featset=os.environ.get("FEATSET", "full"), protocol=proto, test_scenario=s_te, vector=vec[s_te], seed=seed, n_train_scen=len(s_tr))
            tab = {"LogReg": LogisticRegression(max_iter=3000, class_weight="balanced"),
                   "RandomForest": RandomForestClassifier(500, class_weight="balanced_subsample", n_jobs=4, random_state=seed),
                   "XGBoost": XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9,
                                            colsample_bytree=0.9, scale_pos_weight=(ytr[mtr] == 0).sum() / max(1, (ytr[mtr] == 1).sum()),
                                            n_jobs=4, random_state=seed, verbosity=0, eval_metric="logloss")}
            for nm, m in tab.items():
                t0 = time.time(); m.fit(Xtr[mtr], ytr[mtr])
                sv, st = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
                rows.append(dict(**base, model=nm, time_s=time.time() - t0, **score(yte, st, thr_from_val(yva[yva >= 0], sv[yva >= 0]))))
            d = Gtr[0][0].shape[1]
            specs = [("MLP (no graph)", lambda: MLP(d), "none"), ("GCN (k-NN graph)", lambda: GNN("GCN", d), "knn"),
                     ("GCN (provenance)", lambda: GNN("GCN", d), "prov"), ("GraphSAGE (provenance)", lambda: GNN("SAGE", d), "prov"),
                     ("GAT (provenance)", lambda: GNN("GAT", d), "prov")]
            for nm, make, gt in specs:
                def view(G):
                    if gt == "prov": return [(g[0], g[1], g[2]) for g in G]
                    Xp_ = [g[0][: g[3]] for g in G]
                    return [(x, knn_ei(x) if gt == "knn" else np.zeros((2, 0), int), g[2]) for x, g in zip(Xp_, G)]
                t0 = time.time()
                sv, st, info = train_nn(make, view(Gtr), view(Gva), view(Gte), seed)
                yv = np.concatenate([g[2] for g in Gva]); sv = np.concatenate(sv)
                rows.append(dict(**base, model=nm, time_s=time.time() - t0, **info,
                                 **score(yte, st[0], thr_from_val(yv[yv >= 0], sv[yv >= 0]))))
            print(proto, s_te, seed, {r["model"]: round(r["auc"], 3) for r in rows[-8:]}, flush=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    json.dump({"vector": vec, "features": feat}, open(os.path.join(OUT, "atlas_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
