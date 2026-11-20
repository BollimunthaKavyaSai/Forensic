"""Leakage-controlled feature construction for both datasets (revision R3)."""
import numpy as np, pandas as pd, hashlib

WIN_LEAK_COLS = ["Anomaly_Type", "Label"]  # targets; never used as inputs

def load_windows(path, with_leaky=False):
    w = pd.read_csv(path)
    ts = pd.to_datetime(w["Timestamp"])
    X = pd.DataFrame(index=w.index)
    X["User_ID"] = w["User_ID"]
    X["Hour"] = ts.dt.hour
    X["DayOfWeek"] = ts.dt.dayofweek
    X["IsWeekend"] = (X["DayOfWeek"] >= 5).astype(int)
    X["Login_Attempts"] = w["Login_Attempts"].fillna(0)
    X["Login_Info_Missing"] = w["Login_Attempts"].isna().astype(int)
    X["File_Size"] = w["File_Size"].fillna(0)
    X["File_Size_Log"] = np.log1p(X["File_Size"])
    X["File_Info_Missing"] = w["File_Size"].isna().astype(int)
    X["Action_Missing"] = w["Action"].isna().astype(int)
    oct_ = w["IP_Address"].str.split(".", expand=True).astype(int)
    for i in range(4):
        X[f"IP{i+1}"] = oct_[i]
    X["IsPrivateIP"] = ((oct_[0] == 10) | ((oct_[0] == 172) & oct_[1].between(16, 31)) |
                        ((oct_[0] == 192) & (oct_[1] == 168))).astype(int)
    res_dir = w["Resource_Accessed"].str.extract(r"^(/[^/]+/[^/]+)")[0].fillna("other")
    res_file = w["Resource_Accessed"].str.split("/").str[-1]
    cats = {"Activity": w["Activity_Type"], "Action": w["Action"].fillna("None"),
            "ResDir": res_dir, "ResFile": res_file, "FileName": w["File_Name"].fillna("None")}
    for k, v in cats.items():
        X = pd.concat([X, pd.get_dummies(v, prefix=k).astype(int)], axis=1)
    y_bin = (w["Label"] == "Suspicious").astype(int).values
    y_multi = pd.Categorical(w["Anomaly_Type"].fillna("Normal"),
                             categories=["Normal", "Brute_Force", "DDoS_Attempt", "Data_Exfil", "USB_Access"]).codes
    if with_leaky:  # reproduces the columns used in the original notebook
        X["Anomaly_Missing"] = w["Anomaly_Type"].isna().astype(int)
        X["Anomaly_Label"] = y_bin
    return X.astype(float), y_bin, np.asarray(y_multi)

def load_adroit(path):
    a = pd.read_csv(path, sep=";")
    y = (a["LABEL"] == "malware").astype(int).values
    num = a.select_dtypes(include=["int64", "float64"]).replace([np.inf, -np.inf], np.nan)
    num = num.dropna(axis=1, how="all")
    num = num.loc[:, num.var() > 0]
    # duplicate-aware groups: identical feature vectors share a group id
    h = pd.util.hash_pandas_object(num.fillna(-999), index=False).values
    groups = pd.factorize(h)[0]
    return num, y, groups
