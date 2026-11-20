"""
Revision R4: end-to-end functional validation of the prototype's Windows extraction on a real forensic image with
published ground truth (NIST CFReDS "Data Leakage Case", Windows 7 SP1, EnCase E01, 2015).

The prototype's own artefact modules (registry_artifacts, browser_artifacts, recycle_bin, event_logs) are run
unchanged on the image through a pure-Python pytsk3/pyewf stand-in (tsk_shim.py). The new .evtx module
(prototype_ext/evtx_artifacts.py) is evaluated separately. Each output is scored against the NIST answer key.
Usage: python cfreds_validate.py <repo path>
"""
import os, sys, json, time, re, importlib.util
from urllib.parse import unquote_plus, urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import tsk_shim; tsk_shim.install()
REPO = sys.argv[1]; sys.path.insert(0, os.path.join(REPO, "extraction"))
IMG = os.path.join(HERE, "..", "data_real", "cfreds", "cfreds_2015_data_leakage_pc.E01")

# ----------------------------------------------------------------- NIST answer key (selected, machine-checkable)
GT = {
    "computer_name": ["INFORMANT-PC"],
    "timezone": ["Eastern Standard Time"],
    "last_logged_user": ["informant"],
    "ip_address": ["10.11.11.129"], "dhcp_server": ["10.11.11.254"], "default_gateway": ["10.11.11.2"],
    "user_profiles": ["informant", "admin11", "ITechTeam", "temporary"],
    "installed_apps": ["Microsoft Office Professional Plus 2013", "Google Chrome", "Google Update Helper",
                       "Apple Application Support", "Bonjour", "Apple Software Update", "Google Drive",
                       "Microsoft .NET Framework 4", "Eraser"],
    "usb_serials": ["4C530012450531101593", "4C530012550531106501"],
    "executed_programs": ["IE11-Windows6.1-x64-en-us.exe", "GoogleUpdateSetup.exe", "icloudsetup.exe", "EXCEL.EXE",
                          "POWERPNT.EXE", "SOLITAIRE.EXE", "StikyNot.exe", "CHROME.EXE", "OUTLOOK.EXE", "wmplayer.exe",
                          "Eraser 6.2.0.2962.exe", "ccsetup504.exe", "Eraser.exe", "CCLEANER64.EXE",
                          "GOOGLEDRIVESYNC.EXE", "iexplore.exe", "WINWORD.EXE", "xpsrchvw.exe", "iCloud.exe",
                          "dotNetFx40_Full_setup.exe", "googledrivesync.exe", "GoogleUpdate.exe"],
    "search_keywords": ["data leakage methods", "leaking confidential information", "information leakage cases",
                        "intellectual property theft", "how to leak a secret", "file sharing and tethering", "DLP DRM",
                        "e-mail investigation", "Forensic Email Investigation", "what is windows system artifacts",
                        "investigation on windows machine", "windows event logs", "cd burning method",
                        "cd burning method in windows", "external device and forensics", "cloud storage",
                        "digital forensics", "how to delete data", "anti-forensics", "system cleaner",
                        "how to recover data", "data recovery tools", "apple icloud", "google drive"],
    "recycle_bin_note": "Recycle Bin was emptied by the suspect; $I/$R entries require deleted-file recovery",
}
# Q12: security-log start-up / logon / logoff / shutdown events between 09:00 and 18:00 local (UTC-4 with DST)
GT_Q12 = [("2015-03-22 10:51:14", 4608), ("2015-03-22 11:00:08", 4624), ("2015-03-22 11:22:54", 4624),
          ("2015-03-22 12:00:08", 4647), ("2015-03-22 12:00:09", 1100), ("2015-03-23 13:24:23", 4624),
          ("2015-03-23 13:24:23", 4608), ("2015-03-23 14:36:07", 4624), ("2015-03-23 16:00:22", 4624),
          ("2015-03-23 16:01:02", 4624), ("2015-03-23 17:02:53", 4647), ("2015-03-23 17:02:59", 1100),
          ("2015-03-24 09:21:29", 4624), ("2015-03-24 09:21:29", 4608), ("2015-03-24 09:23:40", 4624),
          ("2015-03-24 11:14:30", 4624), ("2015-03-24 11:22:39", 4624), ("2015-03-24 11:46:14", 4624),
          ("2015-03-24 14:28:38", 4624), ("2015-03-24 16:58:52", 4624), ("2015-03-24 17:07:25", 4647),
          ("2015-03-24 17:07:26", 1100), ("2015-03-25 09:05:41", 4624), ("2015-03-25 09:05:41", 4608),
          ("2015-03-25 09:07:49", 4624), ("2015-03-25 09:23:59", 4624), ("2015-03-25 10:31:53", 4624),
          ("2015-03-25 10:45:59", 4637), ("2015-03-25 10:50:28", 4624), ("2015-03-25 10:50:30", 4624),
          ("2015-03-25 10:50:50", 4624), ("2015-03-25 10:56:55", 4624), ("2015-03-25 10:57:18", 4624),
          ("2015-03-25 11:18:54", 4624), ("2015-03-25 11:30:57", 4647), ("2015-03-25 11:31:00", 1100)]


def norm(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())


def recall(found_strings, truth):
    blob = [norm(x) for x in found_strings]
    hit = [t for t in truth if any(norm(t) in b for b in blob)]
    return dict(found=len(hit), total=len(truth), recall=len(hit) / len(truth), missed=[t for t in truth if t not in hit])


def flatten(o, out):
    if isinstance(o, dict):
        for v in o.values(): flatten(v, out)
    elif isinstance(o, (list, tuple)):
        for v in o: flatten(v, out)
    elif o is not None:
        out.append(str(o))
    return out


def keywords_from_urls(urls):
    kws = []
    for u in urls:
        try:
            p = urlparse(u); q = parse_qs(p.query); q.update(parse_qs(p.fragment))
            for k in ("q", "pq"):
                for v in q.get(k, []): kws.append(unquote_plus(v))
        except Exception: pass
    return kws


def main():
    from forensic_extractor import ForensicExtractor
    import pytsk3
    if os.environ.get("FIX_PARTITION") == "1":
        # R4 fix: the released code mounts the first NTFS partition, which on Windows 7 is the 100 MB System
        # Reserved volume. Select the NTFS partition that contains a \Windows directory instead.
        def _find(self):
            for part in pytsk3.Volume_Info(self.img):
                if b"NTFS" not in part.desc: continue
                try:
                    pytsk3.FS_Info(self.img, offset=part.start * 512).open_dir("/Windows"); self.ntfs_offset = part.start * 512
                    print("Found Windows NTFS partition at offset:", self.ntfs_offset); return
                except Exception: continue
        ForensicExtractor._find_ntfs_partition = _find
    t0 = time.time(); ex = ForensicExtractor(IMG); ok = ex.open_image(); t_open = time.time() - t0
    report = {"image_opened": bool(ok), "open_seconds": t_open}
    profiles = ex.discover_user_profiles(); report["profiles"] = profiles
    timings, outputs = {}, {}
    for nm, fn in [("registry", ex.extract_registry_artifacts), ("browser", lambda: ex.extract_browser_artifacts(profiles)),
                   ("recycle_bin", ex.extract_recycle_bin_artifacts), ("event_logs", ex.extract_event_log_artifacts),
                   ("filesystem", ex.extract_filesystem_artifacts)]:
        t = time.time()
        try: outputs[nm] = fn()
        except Exception as e: outputs[nm] = {"error": repr(e)}
        timings[nm] = time.time() - t
    report["module_seconds"] = timings
    json.dump(outputs, open(os.path.join(HERE, "results", "cfreds_prototype_output%s.json" % os.environ.get("TAG", "")), "w"), indent=1, default=str)
    reg = flatten(outputs.get("registry", {}), []); br = flatten(outputs.get("browser", {}), [])
    allstr = flatten(outputs, [])
    urls = [s for s in br if s.startswith("http")]
    sc = {}
    fsys = flatten(outputs.get("filesystem", {}), [])
    for k in ["computer_name", "timezone", "last_logged_user", "ip_address", "dhcp_server", "default_gateway",
              "installed_apps", "usb_serials"]:
        sc[k] = recall(reg, GT[k])
    sc["executed_programs"] = recall(reg + fsys, GT["executed_programs"])  # UserAssist (registry) + Prefetch (filesystem)
    sc["user_profiles"] = recall(profiles, GT["user_profiles"])
    sc["search_keywords"] = recall(keywords_from_urls(urls) + br, GT["search_keywords"])
    sc["browser_urls_extracted"] = len(set(urls))
    rb = outputs.get("recycle_bin", {}); sc["recycle_bin_items"] = len(flatten(rb, []))
    ev = outputs.get("event_logs", {}); sc["event_logs_parsed_by_released_module"] = len(flatten(ev, []))
    # ---- new evtx module
    from prototype_ext.evtx_artifacts import EvtxArtifacts
    t = time.time(); tl = EvtxArtifacts(ex.fs).logon_timeline(); timings["evtx_extension"] = time.time() - t
    import datetime as dt
    local = set()
    for e in (tl if isinstance(tl, list) else []):
        u = dt.datetime.fromisoformat(e["time_utc"].replace("Z", "").split(".")[0])
        local.add(((u - dt.timedelta(hours=4)).strftime("%Y-%m-%d %H:%M:%S"), e["event_id"]))
    hit = [g for g in GT_Q12 if g in local or any(abs((dt.datetime.fromisoformat(g[0]) - dt.datetime.fromisoformat(l[0])).total_seconds()) <= 1 and g[1] == l[1] for l in local)]
    in_window = {l for l in local if "09:00:00" <= l[0][11:] <= "18:00:00"}
    sc["evtx_logon_timeline"] = dict(found=len(hit), total=len(GT_Q12), recall=len(hit) / len(GT_Q12),
                                     n_events_in_window=len(in_window),
                                     missed=[g for g in GT_Q12 if g not in hit], n_events_extracted=len(local))
    report["scores"] = sc; report["module_seconds"] = timings
    json.dump(report, open(os.path.join(HERE, "results", "cfreds_validation%s.json" % os.environ.get("TAG", "")), "w"), indent=1, default=str)
    print(json.dumps(report, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
