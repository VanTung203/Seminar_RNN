from data import train_loader, test_loader, vocab # Import từ file data.py
from model import RNNModel # Import từ file model.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import json
import time

# Xác định thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Sử dụng thiết bị: {device}")

# --- Hàm huấn luyện và đánh giá ---
def train_and_evaluate(model, train_loader, test_loader, epochs=10, lr=0.01, weight_decay=0): # Đề bài không nói rõ weight_decay
    # 1. Khởi tạo loss function và optimizer SGD (không dùng Adam)
    criterion = nn.CrossEntropyLoss() # Phù hợp cho phân loại đa lớp
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.to(device) # Chuyển mô hình sang thiết bị

    print(f"\nBắt đầu huấn luyện với learning rate: {lr}, weight_decay: {weight_decay}")
    for epoch in range(epochs):
        start_time_epoch = time.time()
        model.train() # Đặt mô hình ở chế độ huấn luyện
        running_loss = 0.0
        total_samples_epoch = 0

        for texts, labels in train_loader:
            texts = texts.to(device)
            labels = labels.to(device)

            # Xóa gradients cũ
            optimizer.zero_grad()

            # Forward pass
            predictions = model(texts) # Logits

            # Tính loss
            loss = criterion(predictions, labels)

            # Backward pass
            loss.backward()

            # Cập nhật trọng số
            optimizer.step()

            running_loss += loss.item() * texts.size(0) # Cộng dồn loss của batch
            total_samples_epoch += texts.size(0)

        epoch_loss = running_loss / total_samples_epoch
        epoch_duration = time.time() - start_time_epoch
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Thời gian: {epoch_duration:.2f}s")

    # --- Đánh giá mô hình ---
    model.eval() # Đặt mô hình ở chế độ đánh giá
    all_predictions_list = []
    all_true_labels_list = []

    with torch.no_grad(): # Không cần tính gradient khi đánh giá
        for texts, labels in test_loader:
            texts = texts.to(device)
            labels = labels.to(device)

            outputs = model(texts) # Logits
            # Lấy nhãn dự đoán (lớp có xác suất/logit cao nhất)
            _, predicted_labels = torch.max(outputs, 1) # Lấy argmax dọc theo chiều lớp

            all_predictions_list.extend(predicted_labels.cpu().tolist())
            all_true_labels_list.extend(labels.cpu().tolist())

    acc = accuracy_score(all_true_labels_list, all_predictions_list)
    f1 = f1_score(all_true_labels_list, all_predictions_list, average='macro') # Theo yêu cầu đề bài
    return acc, f1

# --- Thử nghiệm Pretrained vs Scratch ---
results = {}

# Tham số mô hình theo đề bài
embedding_dim_config = 100
hidden_dim_config = 128
output_dim_config = 3 # Positive, Negative, Neutral

# Tham số huấn luyện
num_epochs_config = 10 # Có thể điều chỉnh
learning_rate_config = 0.01 # Có thể điều chỉnh
# weight_decay_config = 1e-4 # Thử nghiệm với weight decay

# Đường dẫn tới file GloVe
glove_file_path_config = 'glove.6B.100d.txt'

for use_pretrained_embedding in [True, False]: # Chạy cả hai trường hợp
    experiment_key = f"RNN_Pretrained={use_pretrained_embedding}"
    print(f"\n--- Thử nghiệm: {experiment_key} ---")

    # Khởi tạo mô hình
    # Khi use_pretrained_embedding=True, model.py sẽ cố gắng load GloVe
    model_instance = RNNModel(vocab_size=len(vocab), # Lấy từ data.py
                              embedding_dim=embedding_dim_config,
                              hidden_dim=hidden_dim_config,
                              output_dim=output_dim_config,
                              pretrained=use_pretrained_embedding,
                              vocab_for_glove=vocab if use_pretrained_embedding else None, # Chỉ cần vocab nếu dùng pretrained
                              glove_path=glove_file_path_config if use_pretrained_embedding else None,
                              freeze_embedding=True # Thường thì ta đóng băng GloVe ban đầu
                             )

    start_time_experiment = time.time()
    # Huấn luyện và đánh giá
    accuracy, f1_macro = train_and_evaluate(model_instance,
                                            train_loader,
                                            test_loader,
                                            epochs=num_epochs_config,
                                            lr=learning_rate_config)
                                            # weight_decay=weight_decay_config) # Có thể bỏ qua nếu không muốn dùng

    experiment_duration = time.time() - start_time_experiment

    results[experiment_key] = {"Accuracy": accuracy, "F1-score": f1_macro, "Training_Time_Seconds": experiment_duration}
    print(f"Kết quả cho {experiment_key}: Accuracy = {accuracy:.4f}, F1-score (macro) = {f1_macro:.4f}, Thời gian huấn luyện: {experiment_duration:.2f}s")

# Lưu kết quả vào file JSON
output_json_file = "results.json"
with open(output_json_file, "w", encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\nKết quả đã được lưu vào file: {output_json_file}")