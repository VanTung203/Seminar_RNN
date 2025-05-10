# Phân tích Cảm xúc Văn bản Ngắn Tiếng Việt bằng RNN

Dự án này sử dụng Mạng Nơ-ron Hồi quy (RNN) để phân loại cảm xúc (Tích cực, Tiêu cực, Trung tính) từ các đoạn văn bản ngắn tiếng Việt.

## Tổng quan

-   **Mục tiêu:** Xây dựng mô hình RNN đơn giản để dự đoán cảm xúc.
-   **Dữ liệu:** Văn bản ngắn (<50 từ) về công việc/học tập, nhãn: Positive, Negative, Neutral.
-   **Kiến trúc chính:** Embedding Layer -> RNN Layer (có thể một chiều hoặc hai chiều) -> Dense Layer.
-   **Thử nghiệm:** So sánh hiệu suất giữa embedding học từ đầu (Scratch) và embedding tiền huấn luyện (Pretrained GloVe), cũng như giữa RNN một chiều và hai chiều.

## Yêu cầu môi trường

-   Python 3.8+
-   `torch`, `pandas`, `numpy`, `nltk`, `scikit-learn`

Cài đặt: `pip install torch pandas numpy nltk scikit-learn`

## Cấu trúc thư mục

-   `sentiment_data.csv`: Dữ liệu đầu vào (cần tự tạo).
-   `data.py`: Tiền xử lý dữ liệu, tạo DataLoader.
-   `model.py`: Định nghĩa kiến trúc RNN.
-   `train_eval.py`: Huấn luyện, đánh giá, so sánh mô hình.
-   `results.json`: Lưu kết quả thử nghiệm.
-   `glove.6B.100d.txt`: (Tùy chọn) File GloVe.

## Hướng dẫn sử dụng

1.  **Chuẩn bị `sentiment_data.csv`:**
    -   Gồm 2 cột: `text,label`.
    -   Ví dụ: `"Hôm nay tôi đi làm muộn.","Negative"`
    -   Ít nhất 500 mẫu, không có dòng trống.

2.  **(Tùy chọn)** Đặt file `glove.6B.100d.txt` vào thư mục dự án nếu muốn dùng pretrained embedding.

3.  **Chạy huấn luyện và đánh giá:**
    ```bash
    python train_eval.py
    ```
    Kết quả sẽ được lưu vào `results.json`.
