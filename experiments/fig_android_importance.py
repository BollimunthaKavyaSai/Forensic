"""Illustrative Random Forest (Gini) importance on the full ADROIT feature matrix (top 20)."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from data_prep import load_adroit
num, y, _ = load_adroit(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "adroit_dataset.csv"))
X = num.fillna(num.median())
rf = RandomForestClassifier(300, n_jobs=2, random_state=0).fit(X, y)
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
imp.to_csv("results/android_rf_importance_full.csv")
top = imp.head(20)[::-1]
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7.2, 4.4))
labels = [c.replace("android.permission.", "").replace("com.android.launcher.permission.", "launcher.") for c in top.index]
ax.barh(labels, top.values, color="#2a78d6", height=0.7)
ax.set_xlabel("Mean decrease in Gini impurity"); ax.grid(axis="x", color="#e6e6e6", lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("results/figures/fig_android_importance.png", dpi=300); print(imp.head(25))
