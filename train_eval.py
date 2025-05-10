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

# --- Hàm huấn luyện và đánh giá (giữ nguyên logic bên trong) ---
def train_and_evaluate(model, train_loader, test_loader, epochs=10, lr=0.01, weight_decay=0):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)

    print(f"\nBắt đầu huấn luyện với learning rate: {lr}, weight_decay: {weight_decay}")
    for epoch in range(epochs):
        start_time_epoch = time.time()
        model.train()
        running_loss = 0.0
        total_samples_epoch = 0
        for texts, labels in train_loader:
            texts = texts.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            predictions = model(texts)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * texts.size(0)
            total_samples_epoch += texts.size(0)
        epoch_loss = running_loss / total_samples_epoch
        epoch_duration = time.time() - start_time_epoch
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Thời gian: {epoch_duration:.2f}s")

    model.eval()
    all_predictions_list = []
    all_true_labels_list = []
    with torch.no_grad():
        for texts, labels in test_loader:
            texts = texts.to(device)
            labels = labels.to(device)
            outputs = model(texts)
            _, predicted_labels = torch.max(outputs, 1)
            all_predictions_list.extend(predicted_labels.cpu().tolist())
            all_true_labels_list.extend(labels.cpu().tolist())
    acc = accuracy_score(all_true_labels_list, all_predictions_list)
    f1 = f1_score(all_true_labels_list, all_predictions_list, average='macro')
    return acc, f1

# --- Thử nghiệm Pretrained vs Scratch, Unidirectional vs Bidirectional ---
results = {}

embedding_dim_config = 100
hidden_dim_config = 128 # hidden_dim cho mỗi chiều của RNN
output_dim_config = 3
num_rnn_layers_config = 1 # Theo yêu cầu "mô hình đơn giản", thường là 1 lớp RNN

num_epochs_config = 10
learning_rate_config = 0.01
glove_file_path_config = 'glove.6B.100d.txt' # Cần có file này

# Thêm vòng lặp cho is_bidirectional
for is_bidirectional_config in [False, True]: # Thử nghiệm cả một chiều và hai chiều
    for use_pretrained_embedding in [True, False]:
        direction_str = "Bidirectional" if is_bidirectional_config else "Unidirectional"
        experiment_key = f"RNN_{direction_str}_Pretrained={use_pretrained_embedding}"
        print(f"\n--- Thử nghiệm: {experiment_key} ---")

        model_instance = RNNModel(
            vocab_size=len(vocab),
            embedding_dim=embedding_dim_config,
            hidden_dim=hidden_dim_config, # Sẽ là 128 cho mỗi chiều nếu bidirectional
            output_dim=output_dim_config,
            pretrained=use_pretrained_embedding,
            vocab_for_glove=vocab if use_pretrained_embedding else None,
            glove_path=glove_file_path_config if use_pretrained_embedding else None,
            freeze_embedding=True,
            is_bidirectional=is_bidirectional_config, # Truyền cờ
            num_rnn_layers=num_rnn_layers_config
        )

        start_time_experiment = time.time()
        accuracy, f1_macro = train_and_evaluate(
            model_instance,
            train_loader,
            test_loader,
            epochs=num_epochs_config,
            lr=learning_rate_config
        )
        experiment_duration = time.time() - start_time_experiment

        results[experiment_key] = {
            "Accuracy": accuracy,
            "F1-score": f1_macro,
            "Training_Time_Seconds": experiment_duration,
            "Config": {
                "embedding_dim": embedding_dim_config,
                "hidden_dim_per_direction": hidden_dim_config,
                "num_rnn_layers": num_rnn_layers_config,
                "is_bidirectional": is_bidirectional_config,
                "use_pretrained": use_pretrained_embedding,
                "epochs": num_epochs_config,
                "lr": learning_rate_config
            }
        }
        print(f"Kết quả cho {experiment_key}: Accuracy = {accuracy:.4f}, F1-score (macro) = {f1_macro:.4f}, Thời gian: {experiment_duration:.2f}s")

output_json_file = "results.json"
with open(output_json_file, "w", encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
print(f"\nKết quả đã được lưu vào file: {output_json_file}")