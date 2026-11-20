"""Schematic figures for revision R3 (study design, tabular architectures, prototype architecture)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9})
INK, EDGE = "#222222", "#555555"
FILL = {"data": "#e8f0fb", "proc": "#ffffff", "model": "#fdeee6", "eval": "#e6f6ef", "gap": "#f4f4f4"}


def box(ax, x, y, w, h, text, kind="proc", dashed=False, bold=False, fs=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=FILL[kind],
                                ec=EDGE, lw=0.9, ls="--" if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=INK, fontsize=fs or plt.rcParams["font.size"],
            fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, dashed=False):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=0.9, ls="--" if dashed else "-"))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); return fig, ax


# ---------------- Figure 1: study design
plt.rcParams["font.size"] = 8
fig, ax = canvas(10, 6.2)
ax.text(0.25, 0.97, "A. Benchmark track (all reported metrics)", ha="center", fontweight="bold")
ax.text(0.76, 0.97, "B. Prototype track (functional/performance tests only)", ha="center", fontweight="bold")
box(ax, 0.02, 0.83, 0.21, 0.09, "Simulated activity-log\nCSV (7,400 rows)", "data")
box(ax, 0.26, 0.83, 0.21, 0.09, "ADROIT app-store\nmetadata CSV (11,476 rows)", "data")
box(ax, 0.02, 0.67, 0.21, 0.11, "Leakage audit: remove\ntarget-derived columns;\nlabel-permutation test", "eval")
box(ax, 0.26, 0.67, 0.21, 0.11, "Duplicate audit: identical\nfeature vectors grouped;\nRF selection inside folds", "eval")
box(ax, 0.02, 0.47, 0.21, 0.15, "Tabular baselines\n(Majority, LR, RF, XGBoost)\nMLP (no graph)\nGCN / GAT on k-NN graph", "model")
box(ax, 0.26, 0.47, 0.21, 0.15, "LR, SVM, RF, XGBoost, MLP\nTabTransformer (original)\nFT-Transformer\nDevice-derivable subset", "model")
box(ax, 0.02, 0.27, 0.45, 0.14, "5-fold stratified CV x 3 repeats (15 folds);\nrandom and duplicate-aware grouped splits\n"
    "Mean ± SD; corrected resampled t-test + Wilcoxon; Holm\n"
    "Train/validation/test gaps for every neural model", "eval")
for x in (0.125, 0.365):
    arrow(ax, x, 0.83, x, 0.78); arrow(ax, x, 0.67, x, 0.62); arrow(ax, x, 0.47, x, 0.41)
box(ax, 0.55, 0.83, 0.19, 0.09, "EnCase E01/EWF image", "data")
box(ax, 0.78, 0.83, 0.19, 0.09, "Android adb-backup TAR", "data")
box(ax, 0.55, 0.67, 0.19, 0.11, "pyewf + pytsk3 artefact\nextractors (registry,\nbrowser, NTFS, recycle bin)", "proc")
box(ax, 0.78, 0.67, 0.19, 0.11, "TAR inventory extractor\n(manifests, databases,\nshared prefs, WebView)", "proc")
box(ax, 0.55, 0.47, 0.42, 0.15, "Django REST API + MongoDB case store + React dashboard\n(case management, artefact browsing, timeline, reports)", "proc")
box(ax, 0.55, 0.27, 0.42, 0.14, "Measured: extractor correctness on synthetic TARs with\nknown ground truth; extraction and inference latency", "eval")
arrow(ax, 0.645, 0.83, 0.645, 0.78); arrow(ax, 0.875, 0.83, 0.875, 0.78)
arrow(ax, 0.645, 0.67, 0.645, 0.62); arrow(ax, 0.875, 0.67, 0.875, 0.62); arrow(ax, 0.76, 0.47, 0.76, 0.41)
box(ax, 0.10, 0.04, 0.80, 0.15, "Schema-gap analysis (Section 4.6): which model inputs can be derived from acquired evidence?\n"
    "No validated link between B and A is claimed; metrics in A are NOT end-to-end triage accuracy on acquired evidence.",
    "gap", dashed=True)
arrow(ax, 0.245, 0.27, 0.40, 0.19, dashed=True); arrow(ax, 0.76, 0.27, 0.60, 0.19, dashed=True)
fig.savefig(os.path.join(FIG, "fig1_study_design.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------------- Figure 4: tabular transformer variants
plt.rcParams["font.size"] = 9
fig, ax = canvas(7.2, 3.6)
ax.text(0.24, 0.96, "(a) Repository TabTransformer (single token)", ha="center", fontweight="bold")
ax.text(0.75, 0.96, "(b) FT-Transformer (one token per feature)", ha="center", fontweight="bold")
ys = [0.80, 0.63, 0.46, 0.29, 0.12]
la = ["x in R^51 (standardised)", "Linear embedding 51 -> 128\n(one token)", "2 x Transformer encoder\n(8 heads, d=128, dropout 0.3)",
      "Dropout 0.3", "Linear 128 -> 2, softmax"]
lb = ["x_1 ... x_51 (standardised)", "Per-feature tokens e_j = x_j w_j + b_j\n(51 tokens, d=32) + [CLS]",
      "2 x pre-norm Transformer encoder\n(4 heads, d=32, FFN 64, dropout 0.1)", "[CLS] -> LayerNorm -> ReLU", "Linear 32 -> 2, softmax"]
for col, labels in ((0.06, la), (0.57, lb)):
    for i, (y, t) in enumerate(zip(ys, labels)):
        box(ax, col, y, 0.37, 0.12, t, "model" if 0 < i < 4 else "proc")
        if i: arrow(ax, col + 0.185, ys[i - 1], col + 0.185, y + 0.12)
ax.text(0.245, 0.02, "Sequence length 1: attention weights are identically 1,\nso the block acts as a feed-forward network.",
        ha="center", fontsize=8, style="italic")
ax.text(0.755, 0.02, "Self-attention operates across the 51 feature tokens,\nso feature interactions are modelled explicitly.",
        ha="center", fontsize=8, style="italic")
fig.savefig(os.path.join(FIG, "fig4_tabular_models.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------------- Figure 8: prototype software architecture
plt.rcParams["font.size"] = 8
fig, ax = canvas(10, 5)
box(ax, 0.02, 0.72, 0.28, 0.2, "React 18 + MUI dashboard\nCases, Artefacts, Timeline,\nAnalytics, Anomaly, Reports", "proc")
box(ax, 0.36, 0.72, 0.28, 0.2, "Django REST Framework API\nCase/Job/Note/Audit viewsets;\nextract-raw, ml-infer actions", "proc")
box(ax, 0.70, 0.72, 0.28, 0.2, "Persistence\nPostgreSQL/SQLite: case metadata, audit log\nMongoDB: extracted artefacts, ML output", "data")
box(ax, 0.02, 0.38, 0.30, 0.24, "Windows extraction\npyewf (E01) -> pytsk3 (NTFS)\nregistry, browser history/cookies/\ndownloads, recycle bin, NTFS\nmetadata, legacy .evt logs", "proc")
box(ax, 0.35, 0.38, 0.30, 0.24, "Android extraction\nadb-backup TAR inventory:\npackages, manifests, app DBs,\nshared preferences, WebView,\nSMS backups", "proc")
box(ax, 0.68, 0.38, 0.30, 0.24, "Inference modules (prototype)\nGAT on k-NN graph of case events\nTabTransformer on ADROIT-schema CSV\n(not validated on acquired evidence)", "model", dashed=True)
box(ax, 0.02, 0.06, 0.96, 0.22, "Observed limitations (Section 4.6): .evtx logs are enumerated but not parsed; the released GAT checkpoint expects a\n"
    "target-derived input column that the application fills with six heuristic rules; the scaler is refitted on each case;\n"
    "Android inference requires an ADROIT-schema CSV that the TAR extractor does not produce. These are reported, not hidden.",
    "gap", dashed=True)
arrow(ax, 0.30, 0.82, 0.36, 0.82); arrow(ax, 0.36, 0.78, 0.30, 0.78); arrow(ax, 0.64, 0.82, 0.70, 0.82)
arrow(ax, 0.45, 0.72, 0.17, 0.62); arrow(ax, 0.50, 0.72, 0.50, 0.62); arrow(ax, 0.55, 0.72, 0.83, 0.62)
fig.savefig(os.path.join(FIG, "fig8_prototype_architecture.png"), dpi=300, bbox_inches="tight"); plt.close(fig)
print("ok")
