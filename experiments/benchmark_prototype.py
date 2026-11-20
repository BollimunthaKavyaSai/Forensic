"""
Revision R3: functional test and performance measurement of prototype components.
 (1) AndroidTarExtractor on synthetic adb-backup TARs with known ground truth (correctness + throughput)
 (2) Deployed Windows inference path (kNN graph + released GAT checkpoint) latency vs. number of events
 (3) Android TabTransformer (repository architecture) inference latency vs. number of apps
Usage: python benchmark_prototype.py <path to cloned computer_forensic repo>
"""
import sys, os, io, time, json, tarfile, tempfile
import numpy as np, torch
sys.path.insert(0, sys.argv[1])
import importlib.util
_spec = importlib.util.spec_from_file_location('android_tar_extractor', os.path.join(sys.argv[1], 'extraction', 'android_tar_extractor.py'))
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m); AndroidTarExtractor = _m.AndroidTarExtractor
from ai_ml.model_infer import GAT, _build_knn_graph
from ai_ml.android_model_infer import TabTransformer
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(4)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
res = {"android_tar": [], "windows_gat_inference": [], "android_tabtransformer_inference": []}


def add(tf, name, size=64):
    ti = tarfile.TarInfo(name); data = os.urandom(size); ti.size = len(data); tf.addfile(ti, io.BytesIO(data))


# (1) synthetic adb-backup layout: apps/<pkg>/_manifest, apps/<pkg>/db/*.db, apps/<pkg>/sp/*.xml, webview
for n_pkg in [50, 200, 1000, 5000]:
    rng = np.random.RandomState(n_pkg); truth = {"packages": n_pkg, "manifests": 0, "db": 0, "sp": 0, "webview": 0}
    path = os.path.join(tempfile.gettempdir(), f"synthetic_android_{n_pkg}.tar")
    with tarfile.open(path, "w") as tf:
        for i in range(n_pkg):
            p = f"com.synthetic.app{i}"
            add(tf, f"apps/{p}/_manifest"); truth["manifests"] += 1
            for j in range(rng.randint(0, 4)): add(tf, f"apps/{p}/db/data{j}.db", 4096); truth["db"] += 1
            for j in range(rng.randint(0, 3)): add(tf, f"apps/{p}/sp/prefs{j}.xml"); truth["sp"] += 1
            if rng.rand() < 0.2: add(tf, f"apps/{p}/r/app_webview/Cookies"); truth["webview"] += 1
            add(tf, f"apps/{p}/f/cache.bin")
    t0 = time.perf_counter()
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        ex = AndroidTarExtractor(path); a = ex.extract_all_artifacts(); ex.close()
    dt = time.perf_counter() - t0
    got = {"packages": len(a["android_packages"]), "manifests": len(a["manifests"]), "db": len(a["app_databases"]),
           "sp": len(a["shared_preferences"]), "webview": len(a["webview_artifacts"])}
    res["android_tar"].append({"n_packages": n_pkg, "tar_mb": os.path.getsize(path) / 1e6, "seconds": dt,
                               "members": a["extraction_info"]["total_members"],
                               "all_counts_correct": got == truth, "truth": truth, "extracted": got})
    os.remove(path)
    print("tar", n_pkg, dt, got == truth, flush=True)

# (2) Windows GAT inference with the released checkpoint
sd = torch.load(os.path.join(sys.argv[1], "dl_models", "gat_best.pth"), map_location="cpu")
gat = GAT(20, 16, 5); gat.load_state_dict(sd); gat.eval()
for n in [1000, 5000, 10000, 50000]:
    X = StandardScaler().fit_transform(np.random.RandomState(0).randn(n, 20)).astype(np.float32)
    t = []
    for rep in range(3):
        t0 = time.perf_counter(); ei = _build_knn_graph(X, 8); t1 = time.perf_counter()
        with torch.no_grad(): gat(torch.tensor(X), ei)
        t2 = time.perf_counter(); t.append((t1 - t0, t2 - t1))
    t = np.array(t)
    res["windows_gat_inference"].append({"n_events": n, "knn_graph_s": float(t[:, 0].mean()),
                                         "gat_forward_s": float(t[:, 1].mean()), "total_s": float(t.sum(1).mean())})
    print("gat", n, t.mean(0), flush=True)

# (3) Android TabTransformer (repository architecture, 51 inputs)
tt = TabTransformer(51).eval()
for n in [100, 1000, 10000]:
    X = torch.randn(n, 51); t = []
    for rep in range(3):
        t0 = time.perf_counter()
        with torch.no_grad(): tt(X)
        t.append(time.perf_counter() - t0)
    res["android_tabtransformer_inference"].append({"n_apps": n, "seconds": float(np.mean(t))})
    print("tt", n, np.mean(t), flush=True)

import platform
res["hardware"] = {"cpu": platform.processor(), "threads": torch.get_num_threads(), "torch": torch.__version__,
                   "python": platform.python_version(), "os": platform.platform()}
json.dump(res, open(os.path.join(OUT, "prototype_benchmark.json"), "w"), indent=1)
