"""
Revision R3: behaviour of the DEPLOYED Windows inference path (ai_ml/model_infer.py + released gat_best.pth).
Activity records are mapped into the application's 20-column schema exactly as ai_ml/feature_extractor.py does
(including its heuristic Anomaly_Label rules and the constant Anomaly_Missing = 0), the scaler is refitted per
case as in ai_ml/preprocessing.py, and the released checkpoint is run on a k-NN graph (k = 8).
We measure (i) how often the output agrees with the heuristic rules, (ii) how well it ranks the true labels, and
(iii) how much the output changes when only the heuristic column is altered.
Usage: python deployed_path_test.py <repo path>
"""
import sys, os, json, zlib
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, roc_auc_score
sys.path.insert(0, sys.argv[1])
from ai_ml.model_infer import GAT, _build_knn_graph
from ai_ml.preprocessing import FEATURE_COLUMNS, preprocess_dataframe

torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))
w = pd.read_csv(os.path.join(HERE, "data", "cybercrime_forensic_dataset.csv"))
ts = pd.to_datetime(w.Timestamp)
ACT = {"File_Access": 4, "File_Modification": 4, "File_Deletion": 10, "USB_Insert": 6, "Login": 9, "Remote_Login": 9,
       "Network_Traffic": 8}
ACTION = {"Read": 1, "Write": 2, "Delete": 4, "Success": 12, "Failed": 12, "Inserted": 5}
df = pd.DataFrame({
    "User_ID": w.User_ID, "Activity_Type": w.Activity_Type.map(ACT), "Action": w.Action.map(ACTION).fillna(0),
    "Resource_Accessed": w.Resource_Accessed.apply(lambda s: zlib.crc32(s.encode()) % 10000),
    "Login_Attempts": w.Login_Attempts.fillna(0), "File_Size": w.File_Size.fillna(0), "Hour": ts.dt.hour,
    "File_Info_Missing": w.File_Name.isna().astype(int), "Login_Info_Missing": w.Login_Attempts.isna().astype(int),
    "DayOfWeek": ts.dt.dayofweek})
df["Action_Missing"] = (df.Action == 0).astype(int)
df["Anomaly_Missing"] = 0  # placeholder used by the application
df["IsWeekend"] = (df.DayOfWeek >= 5).astype(int)
o = w.IP_Address.str.split(".", expand=True).astype(int)
for i in range(4): df[f"IP{i+1}"] = o[i]
df["IsPrivateIP"] = ((o[0] == 10) | ((o[0] == 172) & o[1].between(16, 31)) | ((o[0] == 192) & (o[1] == 168))).astype(int)
df["File_Size_Log"] = np.log1p(df.File_Size)


def heuristic(d):  # identical rules to ForensicFeatureExtractor._add_anomaly_labels
    lab = ((d.Hour >= 23) | (d.Hour <= 5) | (d.File_Size > d.File_Size.quantile(0.95)) | (d.Login_Attempts > 3) |
           (d.IsWeekend == 1) | (d.IsPrivateIP == 0) | (d.Activity_Type == 10))
    return lab.astype(int)


y_true = (w.Label == "Suspicious").astype(int).values
sd = torch.load(os.path.join(sys.argv[1], "dl_models", "gat_best.pth"), map_location="cpu")
model = GAT(20, 16, 5); model.load_state_dict(sd); model.eval()


def infer(d):
    X, _, _ = preprocess_dataframe(d.copy())
    ei = _build_knn_graph(X, 8)
    with torch.no_grad():
        P = F.softmax(model(torch.tensor(X, dtype=torch.float32), ei), 1).numpy()
    return P[:, 1:].max(1), (P.argmax(1) != 0).astype(int)  # same scoring as run_gat_inference (5-class head)


rng = np.random.RandomState(0); rows = []
for c in range(20):  # 20 simulated cases of 500 events
    idx = rng.choice(len(df), 500, replace=False); d = df.iloc[idx].reset_index(drop=True)
    d["Anomaly_Label"] = heuristic(d); h = d.Anomaly_Label.values
    score, pred = infer(d)
    d0 = d.copy(); d0["Anomaly_Label"] = 0; s0, p0 = infer(d0)
    d1 = d.copy(); d1["Anomaly_Label"] = 1 - h; s1, p1 = infer(d1)
    yt = y_true[idx]
    rows.append(dict(case=c, heuristic_positive_rate=h.mean(), predicted_anomaly_rate=pred.mean(),
                     agreement_with_heuristic=(pred == h).mean(), kappa_with_heuristic=cohen_kappa_score(pred, h),
                     auc_score_vs_heuristic=roc_auc_score(h, score) if 0 < h.mean() < 1 else np.nan,
                     auc_score_vs_true_label=roc_auc_score(yt, score),
                     predictions_changed_when_heuristic_zeroed=(p0 != pred).mean(),
                     predictions_changed_when_heuristic_inverted=(p1 != pred).mean()))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(HERE, "results", "deployed_path_cases.csv"), index=False)
summary = {k: [float(res[k].mean()), float(res[k].std())] for k in res.columns if k != "case"}
json.dump(summary, open(os.path.join(HERE, "results", "deployed_path_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
