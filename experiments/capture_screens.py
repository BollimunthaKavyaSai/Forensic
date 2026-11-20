"""Capture dashboard screenshots of the locally running prototype (frontend :3000, backend :8000) with headless Edge."""
import json, os, sys, urllib.request
from playwright.sync_api import sync_playwright

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "figures", "screens"); os.makedirs(OUT, exist_ok=True)
cases = json.load(urllib.request.urlopen("http://127.0.0.1:8000/api/cases/"))
cases = cases.get("results", cases)
case = [c for c in sorted(cases, key=lambda c: c["id"]) if c["title"].startswith("CFReDS")][-1]
logcase = [c for c in cases if c["title"].startswith("ATLASv2")][-1]
pages = [("basic-info", "basic_info"), ("artifacts", "artifacts"), ("timeline", "timeline"), ("dashboard", "dashboard"),
]
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True)
    pg = b.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
    pg.goto("http://localhost:3000/login")
    pg.evaluate("""c => { localStorage.setItem('token','demo-token'); localStorage.setItem('username','examiner');
                         localStorage.setItem('selectedCase', JSON.stringify(c)); }""", case)
    for route, name in pages:
        pg.goto(f"http://localhost:3000/{route}"); pg.wait_for_timeout(12000)
        pg.screenshot(path=os.path.join(OUT, f"{name}.png"), full_page=False); print("saved", name)
    if len(sys.argv) > 1 and sys.argv[1] == "infer":
        pg.evaluate("c => localStorage.setItem('selectedCase', JSON.stringify(c))", logcase); case = logcase
        pg.goto(f"http://localhost:3000/cases/{case['id']}/anomaly-detection"); pg.wait_for_timeout(4000)
        pg.get_by_role("button", name="Run Analysis").click(); pg.wait_for_timeout(45000)
        pg.screenshot(path=os.path.join(OUT, "anomaly_after.png"), full_page=True); print("saved anomaly_after")
    b.close()
