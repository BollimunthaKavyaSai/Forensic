"""
Revision R4 extension to the prototype: parse Windows Vista+ .evtx logs (the released event_logs.py lists .evtx
files but decodes only legacy .evt). Uses python-evtx (pure Python). Designed to plug into EventLogArtifacts via the
same pytsk3-style filesystem object used by the other modules.

Outputs per event: channel, event_id, record_id, utc time, computer, and the EventData name/value pairs, which is
what the Windows triage model (atlas track) consumes after feature construction.
"""
import io, re, tempfile, os
from Evtx.Evtx import Evtx

_EID = re.compile(r"<EventID[^>]*>(\d+)</EventID>")
_REC = re.compile(r"<EventRecordID>(\d+)</EventRecordID>")
_TIM = re.compile(r"<TimeCreated SystemTime=\"([^\"]+)\"")
_CMP = re.compile(r"<Computer>([^<]*)</Computer>")
_CHN = re.compile(r"<Channel>([^<]*)</Channel>")
_DAT = re.compile(r"<Data Name=\"([^\"]+)\">(.*?)</Data>", re.S)


def parse_evtx_bytes(raw, event_ids=None, limit=None):
    """Parse a .evtx file given as bytes; returns a list of dicts."""
    fd, tmp = tempfile.mkstemp(suffix=".evtx"); os.write(fd, raw); os.close(fd)
    out = []
    try:
        with Evtx(tmp) as log:
            for rec in log.records():
                try: xml = rec.xml()
                except Exception: continue
                m = _EID.search(xml)
                if not m: continue
                eid = int(m.group(1))
                if event_ids and eid not in event_ids: continue
                t = _TIM.search(xml); r = _REC.search(xml); c = _CMP.search(xml); ch = _CHN.search(xml)
                out.append(dict(event_id=eid, record_id=int(r.group(1)) if r else None,
                                time_utc=t.group(1) if t else None, computer=c.group(1) if c else None,
                                channel=ch.group(1) if ch else None, data=dict(_DAT.findall(xml))))
                if limit and len(out) >= limit: break
    finally:
        os.remove(tmp)
    return out


class EvtxArtifacts:
    LOGDIR = "/Windows/System32/winevt/Logs"

    def __init__(self, fs): self.fs = fs

    def read(self, path):
        f = self.fs.open(path); return f.read_random(0, f.info.meta.size)

    def extract(self, logs=("Security.evtx", "System.evtx"), event_ids=None):
        res = {}
        for name in logs:
            try: res[name] = parse_evtx_bytes(self.read(f"{self.LOGDIR}/{name}"), event_ids)
            except Exception as e: res[name] = {"error": str(e)}
        return res

    def logon_timeline(self):
        """System start/stop and user logon/logoff events (Security log), as used by examiners for Q12-type tasks."""
        ev = self.extract(("Security.evtx",), {4608, 1100, 4624, 4634, 4647, 4637, 4625})["Security.evtx"]
        if isinstance(ev, dict): return ev
        keep = []
        for e in ev:
            d = e["data"]
            keep.append(dict(time_utc=e["time_utc"], event_id=e["event_id"], user=d.get("TargetUserName", ""),
                             logon_type=d.get("LogonType", "")))
        return keep
