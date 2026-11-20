"""
Revision R4: parse ATLASv2 Windows Security audit logs (real Windows 7 hosts, Riddle et al. 2023) into per-scenario
provenance graphs and behavioural process features, and attach REAPr process labels (attack / contaminated / benign).

Nodes: processes, files and remote endpoints.  Edges: parent->child (4688), process->file (4656/4663/4660),
process->endpoint (5156/5158).  Process features are behavioural counts only; executable names and timestamps are
NOT used as features, so a model cannot memorise attack-specific names (e.g. payload.exe) or attack times.
Usage: python atlas_parse.py           (writes results/atlas_graphs.pkl)
"""
import os, re, glob, pickle, collections, ipaddress
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "data_real", "atlasv2")
EV = re.compile(r"<Event [^>]*>(.*?)</Event>", re.S)
EID = re.compile(r"<EventID>(\d+)</EventID>")
DAT = re.compile(r"<Data Name='([^']+)'>(.*?)</Data>", re.S)

DIRS = {"windows": r"c:\windows", "progfiles": r"c:\program files", "downloads": r"\downloads", "appdata": r"\appdata",
        "temp": r"\temp", "desktop": r"\desktop", "documents": r"\documents", "users": r"c:\users"}
EXTS = {"exe": (".exe", ".dll", ".sys"), "doc": (".doc", ".docx", ".xls", ".xlsx", ".rtf", ".ppt", ".pptx"),
        "pdf": (".pdf",), "script": (".bat", ".ps1", ".vbs", ".js", ".hta"), "web": (".htm", ".html", ".php"),
        "tmp": (".tmp",), "archive": (".zip", ".rar", ".7z", ".cab")}


def pcat(path):
    p = path.lower()
    return {"img_system32": int(r"\windows\system32" in p), "img_windows": int(p.startswith(r"c:\windows")),
            "img_progfiles": int(p.startswith(r"c:\program files")),
            "img_userdir": int(p.startswith(r"c:\users")), "img_downloads": int(r"\downloads" in p),
            "img_temp": int(r"\temp" in p or r"\appdata" in p)}


def is_private(ip):
    try: return ipaddress.ip_address(ip).is_private
    except ValueError: return True


def iter_events(path, chunk=64 << 20):
    with open(path, encoding="utf-8", errors="ignore") as f:
        buf = ""
        while True:
            s = f.read(chunk)
            if not s and not buf: break
            buf += s; last = buf.rfind("</Event>")
            if last < 0:
                if not s: break
                continue
            for m in EV.finditer(buf, 0, last + 8):
                body = m.group(1); e = EID.search(body)
                if e: yield int(e.group(1)), dict(DAT.findall(body))
            buf = buf[last + 8:]
            if not s: break


def parse_scenario(xml):
    P = collections.defaultdict(lambda: collections.Counter())  # process key -> counters
    files = collections.defaultdict(set); ends = collections.defaultdict(set); kids = []; images = {}
    for eid, d in iter_events(xml):
        if eid == 4688:
            child = (int(d.get("NewProcessId", "0x0"), 16), d.get("NewProcessName", "").lower())
            ppid = int(d.get("ProcessId", "0x0"), 16)
            P[child]["created_in_window"] += 1; P[child]["elevated"] += int("%%1937" in d.get("TokenElevationType", ""))
            kids.append((ppid, child)); images[child] = child[1]
            continue
        pid_s = d.get("ProcessId") or d.get("ProcessID"); name = (d.get("ProcessName") or d.get("Application") or "").lower()
        if not pid_s: continue
        try: pid = int(pid_s, 16) if pid_s.startswith("0x") else int(pid_s)
        except ValueError: continue
        if name.startswith("\\device\\harddiskvolume"):
            name = "c:\\" + "\\".join(name.split("\\")[3:])
        k = (pid, name); c = P[k]; c[f"e{eid}"] += 1; c["n_events"] += 1; images[k] = name
        if eid in (4656, 4663, 4660, 4658):
            obj = d.get("ObjectName", "").lower()
            if obj and d.get("ObjectType", "File") == "File":
                files[k].add(obj)
                mask = int(d.get("AccessMask", "0x0"), 16) if d.get("AccessMask", "").startswith("0x") else 0
                c["acc_read"] += int(bool(mask & 0x1)); c["acc_write"] += int(bool(mask & 0x2))
                c["acc_append"] += int(bool(mask & 0x4)); c["acc_exec"] += int(bool(mask & 0x20))
                c["acc_delete"] += int(bool(mask & 0x10000)) + int(eid == 4660)
        elif eid in (5156, 5158):
            ip, port = d.get("DestAddress", ""), d.get("DestPort", "")
            if eid == 5156 and ip:
                ends[k].add(f"{ip}:{port}"); c["net_out"] += int("%%14593" in d.get("Direction", ""))
                c["net_in"] += int("%%14592" in d.get("Direction", ""))
                c["net_public"] += int(not is_private(ip))
                try: c["net_highport"] += int(int(port) > 1024)
                except ValueError: pass
    # Sysmon process-creation events (EID 1) add processes, parent links, integrity level and command-line length
    sm = xml.replace("msft-security", "sysmon")
    if os.path.exists(sm):
        for eid, d in iter_events(sm):
            if eid != 1: continue
            try: k = (int(d.get("ProcessId", "0")), d.get("Image", "").lower())
            except ValueError: continue
            c = P[k]; images[k] = k[1]; c["sysmon_seen"] = 1
            il = d.get("IntegrityLevel", ""); c["il_low"] = int(il == "Low"); c["il_high"] = int(il in ("High", "System"))
            c["user_system"] = int("SYSTEM" in d.get("User", "").upper())
            cl = d.get("CommandLine", ""); c["cmd_len"] = len(cl); c["cmd_args"] = len(cl.split())
            c["created_in_window"] = max(c["created_in_window"], 1)
            try: kids.append((int(d.get("ParentProcessId", "0")), k))
            except ValueError: pass
    kids = list(dict.fromkeys(kids))
    # resolve parent keys by pid (most recent image with that pid)
    by_pid = collections.defaultdict(list)
    for k in images: by_pid[k[0]].append(k)
    edges = []
    for ppid, child in kids:
        for pk in by_pid.get(ppid, []):
            edges.append((pk, child)); P[pk]["n_children"] += 1
    rows = []
    for k, c in P.items():
        fs = files.get(k, set()); es = ends.get(k, set())
        r = dict(pid=k[0], image=k[1], **{f: c.get(f, 0) for f in ["n_events", "e4656", "e4663", "e4658", "e4660",
             "e4690", "e5156", "e5158", "acc_read", "acc_write", "acc_append", "acc_exec", "acc_delete", "net_out",
             "net_in", "net_public", "net_highport", "n_children", "created_in_window", "elevated", "sysmon_seen", "il_low", "il_high", "user_system", "cmd_len", "cmd_args"]})
        r["n_files"] = len(fs); r["n_dirs"] = len({f.rsplit("\\", 1)[0] for f in fs}); r["n_endpoints"] = len(es)
        r["n_remote_ips"] = len({e.rsplit(":", 1)[0] for e in es})
        for nm, sub in DIRS.items(): r[f"f_{nm}"] = sum(sub in f for f in fs)
        for nm, ex in EXTS.items(): r[f"x_{nm}"] = sum(f.endswith(ex) for f in fs)
        r.update(pcat(k[1])); rows.append(r)
    return pd.DataFrame(rows), edges, {k: sorted(v) for k, v in files.items()}, {k: sorted(v) for k, v in ends.items()}


def main():
    lab = pd.read_csv(os.path.join(D, "reapr_atlasv2_labels.csv"), skipinitialspace=True)
    lab["scen"] = lab.attack.str.replace("atlasv2/", "", regex=False)
    lab["process_name"] = lab.process_name.str.lower().str.strip()
    out = {}
    kind_sel = os.environ.get("KIND", "attack")
    paths = sorted(glob.glob(os.path.join(D, "atlasv2", "data", kind_sel, "h*", "msft-security", "msft-security-*.xml")))
    for xml in paths:
        kind = "attack" if os.sep + "attack" + os.sep in xml else "benign"
        scen = os.path.basename(xml).replace("msft-security-", "").replace(".xml", "")
        if kind == "benign": scen = "benign-" + scen
        df, edges, files, ends = parse_scenario(xml)
        df["scen"] = scen; df["kind"] = kind
        if kind == "attack":
            L = lab[lab.scen == scen].drop_duplicates(["process_id", "process_name"])
            m = dict(zip(zip(L.process_id, L.process_name), L.label))
            df["label"] = [m.get((p, i), "unlabelled") for p, i in zip(df.pid, df.image)]
        else:
            df["label"] = "benign_day"
        out[scen] = dict(proc=df, edges=edges, files=files, ends=ends, kind=kind)
        vc = df.label.value_counts().to_dict()
        print(scen, len(df), "procs", len(edges), "tree edges", sum(len(v) for v in files.values()), "file links", vc, flush=True)
    pickle.dump(out, open(os.path.join(HERE, "results", "atlas_graphs.pkl" if kind_sel == "attack" else "atlas_graphs_benign.pkl"), "wb"))


if __name__ == "__main__":
    main()
