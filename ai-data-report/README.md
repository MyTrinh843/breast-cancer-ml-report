# Phân loại khối u vú bằng học máy – báo cáo khoa học tự động
 
Dự án nhóm thuộc học phần *Sáng tạo và quản lý nội dung số* – Đại học Cần Thơ.
 
## Giới thiệu
Quy trình tự động: đọc dữ liệu → làm sạch → trực quan hóa → huấn luyện mô hình
phân loại → sinh báo cáo PDF → dashboard tương tác.
 
## Mục tiêu
Kiểm tra xem các đặc trưng hình thái của nhân tế bào có đủ để phân loại khối u
lành tính/ác tính với độ chính xác cao hay không.
 
## Dữ liệu
Breast Cancer Wisconsin (Diagnostic) – UCI Machine Learning Repository
(569 mẫu, 30 đặc trưng). Tệp mẫu: data/breast_cancer.csv.
 
## Phương pháp
- Làm sạch: pandas, numpy (kiểm tra thiếu, trùng, ngoại lai IQR)
- Trực quan hóa: matplotlib, seaborn
- Mô hình: Hồi quy logistic, Random Forest (scikit-learn), chia 80/20, CV 5-fold
- Báo cáo PDF: reportlab; Dashboard: Streamlit
 
## Kết quả
Cả hai mô hình đạt accuracy khoảng 96,5% trên tập kiểm tra.
Xem báo cáo đầy đủ: results/bao_cao.pdf
 
## Cấu trúc thư mục
data/ (dữ liệu), src/ (mã nguồn), results/ (kết quả), app.py (dashboard)
 
## Cách chạy
pip install -r requirements.txt
python src/analysis.py
python src/report.py
streamlit run app.py
 
## Giấy phép
Phát hành theo giấy phép MIT – xem tệp LICENSE.
 
## Nhóm thực hiện
- Họ tên 1 – MSSV
- Họ tên 2 – MSSV
