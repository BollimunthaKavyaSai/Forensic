"""Schematic figures for revision R4: validated end-to-end pipeline and study design."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from draw_diagrams import box, arrow, canvas, FIG  # noqa: reuse R3 drawing helpers

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8})

# ---------------- Figure 1: end-to-end pipeline with validation points
fig, ax = canvas(10, 6.4)
ax.text(0.26, 0.975, "Windows path", ha="center", fontweight="bold", fontsize=9)
ax.text(0.76, 0.975, "Android path", ha="center", fontweight="bold", fontsize=9)
box(ax, 0.03, 0.85, 0.20, 0.08, "EnCase E01/EWF image", "data")
box(ax, 0.27, 0.85, 0.20, 0.08, "Exported/collected\nWindows event logs", "data")
box(ax, 0.53, 0.85, 0.20, 0.08, "adb-backup TAR", "data")
box(ax, 0.77, 0.85, 0.20, 0.08, "APK files recovered\nfrom device", "data")
box(ax, 0.03, 0.67, 0.44, 0.13, "Extraction: EWF + NTFS (Windows partition selected), registry,\nbrowser, prefetch, recycle bin; "
    ".evtx parser (new)\n[validated: NIST CFReDS image vs. official answer key]", "proc")
box(ax, 0.53, 0.67, 0.44, 0.13, "Extraction: TAR inventory; APK manifest and archive\nfeatures (new; androguard)\n"
    "[validated: synthetic TARs; 49 real F-Droid APKs]", "proc")
box(ax, 0.03, 0.47, 0.44, 0.15, "Process provenance graph from Security/Sysmon events\n(parent-child, process-file, process-endpoint)\n"
    "51 behavioural features per process; no names, no timestamps", "proc")
box(ax, 0.53, 0.47, 0.44, 0.15, "Deployable static feature vector\n(permission flags, counts, archive sizes)\n"
    "identical definition in training and deployment", "proc")
box(ax, 0.03, 0.27, 0.44, 0.15, "Ranked processes for examiner review\nGNN (GCN, GraphSAGE, GAT) vs. tabular (LR, RF, XGBoost, MLP)\n"
    "[validated: ATLASv2 real hosts, REAPr labels,\ntrained only on attacks with a different exploit vector]", "model")
box(ax, 0.53, 0.27, 0.44, 0.15, "Malware likelihood per APK\nLR, RF, XGBoost, MLP, FT-Transformer\n"
    "[validated: KronoDroid real-device data; unseen families,\nfuture years; F-Droid false-positive rate]", "model")
for x in (0.13, 0.37): arrow(ax, x, 0.85, x if x < 0.2 else 0.30, 0.80)
for x in (0.63, 0.87): arrow(ax, x, 0.85, x if x < 0.7 else 0.80, 0.80)
for x in (0.25, 0.75): arrow(ax, x, 0.67, x, 0.62); arrow(ax, x, 0.47, x, 0.42)
box(ax, 0.10, 0.05, 0.80, 0.15, "Django REST API + MongoDB case store + React dashboard\n"
    "(case management, artefact browsing, timeline, model outputs, reports)\n"
    "Every model output is a ranking for human review, not a verdict", "eval")
arrow(ax, 0.25, 0.27, 0.40, 0.20); arrow(ax, 0.75, 0.27, 0.60, 0.20)
fig.savefig(os.path.join(FIG, "fig_r4_pipeline.png"), dpi=300, bbox_inches="tight"); plt.close(fig)
print("ok")
