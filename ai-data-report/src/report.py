"""Bước 3: đọc results/metrics.json + hình -> sinh results/bao_cao.pdf"""
import json, os
from datetime import date
import matplotlib
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, PageBreak, KeepTogether)
 
fdir = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
pdfmetrics.registerFont(TTFont("VN", os.path.join(fdir, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("VN-B", os.path.join(fdir, "DejaVuSans-Bold.ttf")))
 
H1 = ParagraphStyle("H1", fontName="VN-B", fontSize=20, leading=26, textColor=colors.HexColor("#1F3864"), spaceAfter=8)
H2 = ParagraphStyle("H2", fontName="VN-B", fontSize=14, leading=18, textColor=colors.HexColor("#2E75B6"), spaceBefore=12, spaceAfter=6)
P = ParagraphStyle("P", fontName="VN", fontSize=11, leading=16, spaceAfter=6)
CAP = ParagraphStyle("CAP", fontName="VN", fontSize=9, leading=12, textColor=colors.grey, alignment=1, spaceAfter=8)
C = ParagraphStyle("C", fontName="VN", fontSize=9.5, leading=13)
CH = ParagraphStyle("CH", fontName="VN-B", fontSize=9.5, leading=13, textColor=colors.white)
 
 
def fig(path, caption, w=11 * cm):
    from PIL import Image as PI
    iw, ih = PI.open(path).size
    return KeepTogether([Image(path, width=w, height=w * ih / iw), Paragraph(caption, CAP)])
 
 
def footer(c, d):
    c.setFont("VN", 8); c.drawCentredString(A4[0] / 2, 1.2 * cm, f"Trang {d.page}")
 
 
def build(out="results/bao_cao.pdf", title="Phân loại khối u vú bằng học máy",
          authors="Nhóm 01 – Đại học Cần Thơ"):
    m = json.load(open("results/metrics.json", encoding="utf-8"))
    i, F = m["info"], "results/figures"
    best = m["best"]; b = m["models"][best]
    story = [
        Paragraph(title.upper(), H1),
        Paragraph(f"{authors} &nbsp;|&nbsp; Ngày lập: {date.today():%d/%m/%Y}", P),
        Paragraph("1. Giới thiệu đề tài", H2),
        Paragraph("Ung thư vú là một trong những bệnh phổ biến nhất ở phụ nữ; phát hiện sớm và phân biệt "
                  "khối u ác tính với lành tính giúp tăng đáng kể khả năng điều trị thành công. "
                  "Đề tài xây dựng quy trình tự động: làm sạch dữ liệu, phân tích, huấn luyện mô hình "
                  "phân loại đơn giản và sinh báo cáo PDF. <b>Câu hỏi nghiên cứu:</b> các đặc trưng hình thái "
                  "của nhân tế bào có đủ để phân loại khối u với độ chính xác cao hay không?", P),
        Paragraph("2. Mô tả dữ liệu", H2),
        Paragraph(f"Dữ liệu: Breast Cancer Wisconsin (Diagnostic) – UCI Machine Learning Repository. "
                  f"Gồm {i['rows_raw']} mẫu, {m['n_features']} đặc trưng số đo từ ảnh nhân tế bào. "
                  f"Nhãn: {', '.join(f'{k} ({v} ca)' for k, v in m['class_counts'].items())}.", P),
    ]
    rows = [[Paragraph(x, CH) for x in ["Kiểm tra chất lượng", "Kết quả"]]]
    for k, v in [("Số dòng ban đầu", i["rows_raw"]), ("Ô thiếu giá trị", i["missing"]),
                 ("Dòng trùng lặp", i["duplicates"]), ("Ô ngoại lai (quy tắc IQR)", i["outlier_cells"]),
                 ("Số dòng sau làm sạch", i["rows_clean"])]:
        rows.append([Paragraph(k, C), Paragraph(str(v), C)])
    t = Table(rows, colWidths=[8 * cm, 4 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E75B6")),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EAF1FB")]),
                           ("GRID", (0, 0), (-1, -1), .4, colors.grey)]))
    story += [t, Paragraph("Bảng 1. Kết quả kiểm tra và làm sạch dữ liệu.", CAP),
              Paragraph("3. Kết quả phân tích", H2),
              fig(f"{F}/class.png", "Hình 1. Phân bố hai lớp chẩn đoán.", 8.5 * cm),
              fig(f"{F}/box.png", "Hình 2. Bán kính trung bình của khối u ác tính thường lớn hơn.", 8.5 * cm),
              fig(f"{F}/corr.png", "Hình 3. Các đặc trưng kích thước có tương quan rất cao với nhau.", 9.5 * cm),
              PageBreak(), Paragraph("4. Mô hình và đánh giá", H2)]
    rows = [[Paragraph(x, CH) for x in ["Mô hình", "Accuracy", "F1", "CV 5-fold"]]]
    for name, r in m["models"].items():
        rows.append([Paragraph(name, C)] + [Paragraph(f"{r[k]:.3f}", C) for k in ("accuracy", "f1", "cv5")])
    t2 = Table(rows, colWidths=[5.5 * cm, 3 * cm, 3 * cm, 3 * cm])
    t2.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E75B6")),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EAF1FB")]),
                            ("GRID", (0, 0), (-1, -1), .4, colors.grey)]))
    story += [Paragraph("Dữ liệu chia 80% huấn luyện / 20% kiểm tra (có phân tầng, random_state=42).", P),
              t2, Paragraph("Bảng 2. So sánh hai mô hình.", CAP),
              fig(f"{F}/cm.png", f"Hình 4. Ma trận nhầm lẫn của mô hình tốt nhất ({best}).", 8 * cm),
              fig(f"{F}/imp.png", "Hình 5. Các đặc trưng quan trọng nhất theo Random Forest.", 9.5 * cm),
              Paragraph("5. Kết luận và đề xuất", H2),
              Paragraph(f"Mô hình <b>{best}</b> đạt accuracy {b['accuracy']:.1%} và F1 {b['f1']:.3f} trên tập kiểm tra. "
                        f"Các đặc trưng ảnh hưởng nhiều nhất: {', '.join(m['top3'])}. "
                        "Kết quả cho thấy đặc trưng hình thái đủ để phân loại tốt trên bộ dữ liệu này.", P),
              Paragraph("<b>Đề xuất:</b> (1) thử thêm mô hình khác và tinh chỉnh siêu tham số; "
                        "(2) kiểm chứng trên dữ liệu bệnh viện thực tế; (3) mô hình chỉ mang tính hỗ trợ, "
                        "không thay thế chẩn đoán của bác sĩ.", P)]
    SimpleDocTemplate(out, pagesize=A4, leftMargin=2.5 * cm, rightMargin=2.5 * cm, topMargin=2.2 * cm,
                      bottomMargin=2 * cm, title=title).build(story, onFirstPage=footer, onLaterPages=footer)
    return out
 
 
if __name__ == "__main__":
    print("Đã tạo", build())
