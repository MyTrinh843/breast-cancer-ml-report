"""Bước 1-2: tải dữ liệu, làm sạch, trực quan hóa, huấn luyện mô hình, lưu kết quả."""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
 
DATA = "data/breast_cancer.csv"
FIG = "results/figures"
plt.rcParams["font.family"] = "DejaVu Sans"
sns.set_theme(style="whitegrid", font="DejaVu Sans")
 
 
def load_data():
    """Dùng bộ Breast Cancer Wisconsin (UCI). Lưu ra CSV để repo có dữ liệu mẫu."""
    if not os.path.exists(DATA):
        d = load_breast_cancer(as_frame=True)
        df = d.frame.copy()                     # 30 đặc trưng + cột target
        df["diagnosis"] = df["target"].map({0: "Ác tính", 1: "Lành tính"})
        df = df.drop(columns="target")
        os.makedirs("data", exist_ok=True)
        df.to_csv(DATA, index=False, encoding="utf-8-sig")
    return pd.read_csv(DATA)
 
 
def clean(df):
    info = {"rows_raw": len(df), "missing": int(df.isna().sum().sum()),
            "duplicates": int(df.duplicated().sum())}
    df = df.drop_duplicates().dropna()
    num = df.select_dtypes("number")
    q1, q3 = num.quantile(.25), num.quantile(.75)
    iqr = q3 - q1
    info["outlier_cells"] = int(((num < q1 - 1.5 * iqr) | (num > q3 + 1.5 * iqr)).sum().sum())
    info["rows_clean"] = len(df)
    return df, info
 
 
def make_figures(df):
    os.makedirs(FIG, exist_ok=True)
    # 1. Cân bằng lớp
    plt.figure(figsize=(5, 3.4))
    ax = sns.countplot(data=df, x="diagnosis", hue="diagnosis", palette=["#C00000", "#2E75B6"], legend=False)
    for c in ax.containers: ax.bar_label(c)
    plt.xlabel("Chẩn đoán"); plt.ylabel("Số ca"); plt.title("Phân bố hai lớp")
    plt.tight_layout(); plt.savefig(f"{FIG}/class.png", dpi=200); plt.close()
    # 2. Boxplot
    plt.figure(figsize=(5, 3.4))
    sns.boxplot(data=df, x="diagnosis", y="mean radius", hue="diagnosis",
                palette=["#C00000", "#2E75B6"], legend=False)
    plt.xlabel("Chẩn đoán"); plt.ylabel("Bán kính trung bình (mm)")
    plt.title("Bán kính khối u theo chẩn đoán")
    plt.tight_layout(); plt.savefig(f"{FIG}/box.png", dpi=200); plt.close()
    # 3. Tương quan
    cols = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness", "mean concavity"]
    plt.figure(figsize=(5.6, 4.4))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="Blues", cbar=False)
    plt.title("Ma trận tương quan (6 đặc trưng)")
    plt.tight_layout(); plt.savefig(f"{FIG}/corr.png", dpi=200); plt.close()
 
 
def train(df):
    X = df.drop(columns="diagnosis")
    y = (df["diagnosis"] == "Ác tính").astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    models = {
        "Hồi quy logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }
    res, best_name, best_acc, best_pred = {}, None, -1, None
    for name, m in models.items():
        m.fit(Xtr, ytr); pred = m.predict(Xte)
        acc = accuracy_score(yte, pred)
        res[name] = {"accuracy": round(acc, 4), "f1": round(f1_score(yte, pred), 4),
                     "cv5": round(cross_val_score(m, X, y, cv=5).mean(), 4)}
        if acc > best_acc:
            best_name, best_acc, best_pred = name, acc, pred
    # Ma trận nhầm lẫn
    cm = confusion_matrix(yte, best_pred)
    plt.figure(figsize=(4.4, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Lành tính", "Ác tính"], yticklabels=["Lành tính", "Ác tính"])
    plt.xlabel("Dự đoán"); plt.ylabel("Thực tế"); plt.title(f"Ma trận nhầm lẫn – {best_name}")
    plt.tight_layout(); plt.savefig(f"{FIG}/cm.png", dpi=200); plt.close()
    # Tầm quan trọng đặc trưng (Random Forest)
    rf = models["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values().tail(8)
    plt.figure(figsize=(5.6, 3.6))
    plt.barh(imp.index, imp.values, color="#ED7D31")
    plt.xlabel("Mức độ quan trọng"); plt.title("Top 8 đặc trưng quan trọng")
    plt.tight_layout(); plt.savefig(f"{FIG}/imp.png", dpi=200); plt.close()
    return res, best_name, list(imp.index[::-1][:3])
 
 
def main():
    df = load_data()
    df, info = clean(df)
    make_figures(df)
    res, best, top3 = train(df)
    out = {"info": info, "models": res, "best": best, "top3": top3,
           "n_features": df.shape[1] - 1,
           "class_counts": df["diagnosis"].value_counts().to_dict()}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
 
 
if __name__ == "__main__":
    main()
