"""
Launch the prototype's Django backend for the R4 evaluation on this machine.
- pytsk3/pyewf native DLLs are blocked by the machine's Application Control policy, so the pure-Python stand-in
  (experiments/tsk_shim.py) is installed first; the prototype's extraction code itself is unchanged.
- No MongoDB server is installed; pymongo.MongoClient is replaced by mongomock's in-memory client.
- PostgreSQL is not installed; settings.py already falls back to SQLite when PostgreSQL is unreachable.
Usage: python run_backend_eval.py [port]
"""
import os, sys
# R4 fix: the app prints non-ASCII symbols (e.g. U+2713); on a Windows cp1252 console this raised UnicodeEncodeError
# after extraction and marked every case as failed. Force UTF-8 output.
for _s in (sys.stdout, sys.stderr): _s.reconfigure(encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"
HERE = os.path.dirname(os.path.abspath(__file__))
# demo case: ATLASv2 host 1, attack S4 (Office exploit); model trained only on the Flash-exploit attacks
os.environ.setdefault("DEMO_SECURITY_XML", os.path.join(HERE, "..", "data_real", "atlasv2", "atlasv2", "data", "attack", "h1",
                                                        "msft-security", "msft-security-h1-s4.xml"))
os.environ.setdefault("WIN_MODEL_FILE", "windows_triage_no_office.pkl")
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))
import tsk_shim; tsk_shim.install()
import mongomock, pymongo
_shared = mongomock.MongoClient()  # one in-memory server shared by storage and retrieval (as a real mongod would be)
pymongo.MongoClient = lambda *a, **k: _shared
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "backend"))
# R4 fix (also evaluated in cfreds_validate.py): mount the NTFS partition that contains \Windows, not the first one
import pytsk3
from extraction.forensic_extractor import ForensicExtractor
def _find(self):
    for part in pytsk3.Volume_Info(self.img):
        if b"NTFS" not in part.desc: continue
        try:
            pytsk3.FS_Info(self.img, offset=part.start * 512).open_dir("/Windows"); self.ntfs_offset = part.start * 512; return
        except Exception: continue
ForensicExtractor._find_ntfs_partition = _find
# R4: route the dashboard's inference endpoints to the validated triage models (experiments/prototype_ext)
from prototype_ext import triage_models as TM
def _win(case_id, threshold=0.5, top_n=50):
    xml = os.environ.get("DEMO_SECURITY_XML")
    if not xml: return {"success": False, "error": "set DEMO_SECURITY_XML to an exported Security log for this case"}
    r = TM.windows_triage(xml, threshold=threshold, top_n=top_n)
    for i, a in enumerate(r["top_anomalies"]):  # field names expected by the dashboard
        a.update(rank=i + 1, anomaly_score=a["score"], activity_type="process", description=a["image"])
    return r
def _android(csv_path, features_path=None, model_path=None, threshold=0.5, top_n=50):
    import glob
    apks = sorted(glob.glob(os.path.join(csv_path, "*.apk"))) if os.path.isdir(csv_path) else [csv_path]
    r = TM.android_triage(apks, threshold=threshold, top_n=top_n)
    for i, a in enumerate(r["top_anomalies"]): a.update(rank=i + 1, anomaly_score=a["score"], app_name=a["package"])
    return r
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "forensic_backend.settings")
sys.path.insert(0, os.path.join(HERE, "backend")); os.chdir(os.path.join(HERE, "backend"))
django.setup()
import forensic_api.views as V
V.run_gat_inference = _win; V.run_android_inference = _android
os.chdir(os.path.join(HERE, "backend"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "forensic_backend.settings")
from django.core.management import execute_from_command_line
port = sys.argv[1] if len(sys.argv) > 1 else "8000"
execute_from_command_line(["manage.py", "migrate", "--noinput"])
execute_from_command_line(["manage.py", "runserver", f"127.0.0.1:{port}", "--noreload"])
