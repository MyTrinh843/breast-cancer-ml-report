"""Dashboard Streamlit: chọn biến, xem biểu đồ, tải báo cáo PDF."""
import os
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
 
from src import analysis, report
 
st.set_page_config(page_title="Phân tích Breast Cancer", page_icon="📊", layout="wide")
plt.rcParams["font.family"] = "DejaVu Sans"
 
 
@st.cache_data
def load():
    return analysis.load_data()
 
 
df = load()
features = [c for c in df.columns if c != "diagnosis"]
 
st.title("📊 Dashboard phân tích dữ liệu Breast Cancer Wisconsin")
st.caption("Nguồn: UCI Machine Learning Repository – 569 mẫu, 30 đặc trưng")
 
# ----- Thanh bên: bộ lọc -----
st.sidebar.header("Bộ lọc")
classes = st.sidebar.multiselect("Chẩn đoán", sorted(df["diagnosis"].unique()),
                                 default=sorted(df["diagnosis"].unique()))
chart = st.sidebar.radio("Loại biểu đồ", ["Histogram", "Boxplot", "Scatter"])
x = st.sidebar.selectbox("Biến X", features, index=0)
y = st.sidebar.selectbox("Biến Y (cho Scatter)", features, index=1)
data = df[df["diagnosis"].isin(classes)]
 
# ----- Các tab -----
tab1, tab2, tab3 = st.tabs(["📄 Dữ liệu", "📈 Biểu đồ", "📑 Báo cáo PDF"])
 
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Số dòng đang chọn", len(data))
    c2.metric("Số đặc trưng", len(features))
    c3.metric("Ô thiếu giá trị", int(data.isna().sum().sum()))
    st.dataframe(data.head(100), use_container_width=True)
    st.subheader("Thống kê mô tả")
    st.dataframe(data[features].describe().T.round(3), use_container_width=True)
 
with tab2:
    if data.empty:
        st.warning("Hãy chọn ít nhất một nhóm chẩn đoán.")
    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        if chart == "Histogram":
            sns.histplot(data=data, x=x, hue="diagnosis", kde=True, ax=ax)
        elif chart == "Boxplot":
            sns.boxplot(data=data, x="diagnosis", y=x, hue="diagnosis", legend=False, ax=ax)
        else:
            sns.scatterplot(data=data, x=x, y=y, hue="diagnosis", ax=ax)
        st.pyplot(fig)
        plt.close(fig)
 
with tab3:
    st.write("Chạy lại phân tích + mô hình và sinh báo cáo PDF mới.")
    if st.button("🔄 Tạo báo cáo"):
        with st.spinner("Đang phân tích và dựng PDF..."):
            analysis.main()
            report.build()
        st.success("Đã tạo xong!")
    if os.path.exists("results/bao_cao.pdf"):
        with open("results/bao_cao.pdf", "rb") as f:
            st.download_button("⬇️ Tải báo cáo PDF", f, file_name="bao_cao.pdf", mime="application/pdf")
