import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
import nltk
from sklearn.model_selection import train_test_split
from collections import Counter
nltk.download('punkt')

# --- Đọc dữ liệu từ CSV ---
data = pd.read_csv('sentiment_data.csv').dropna() # Đọc và loại bỏ dòng trống
texts = data['text'].tolist()
# Ánh xạ nhãn chữ sang số: Positive -> 0, Negative -> 1, Neutral -> 2
labels = data['label'].map({'Positive': 0, 'Negative': 1, 'Neutral': 2}).tolist()

# --- Tokenize và xây dựng từ điển ---
tokenized_texts = [word_tokenize(t.lower()) for t in texts]
all_words = [w for txt in tokenized_texts for w in txt]
# Đề bài yêu cầu vocab_size x 100D, số từ phổ biến nên cân nhắc để vocab_size không quá lớn
# Với embedding 100D, 4998 từ phổ biến + 2 token đặc biệt là 5000, hợp lý.
most_common = Counter(all_words).most_common(4998)
vocab = {'<PAD>': 0, '<UNK>': 1} # PAD là 0, UNK là 1
for i, (w, _) in enumerate(most_common, 2): # Bắt đầu từ index 2
    vocab[w] = i
vocab_size = len(vocab)

# --- Hàm chuyển đổi tokens thành indices và padding ---
# Đảm bảo max_len_text khớp với yêu cầu văn bản ngắn < 50 từ
max_len_text = 50
def to_indices(tokens, max_len):
    idxs = [vocab.get(t, vocab['<UNK>']) for t in tokens][:max_len] # Dùng vocab['<UNK>'] nếu từ không có
    return idxs + [vocab['<PAD>']] * (max_len - len(idxs)) # Dùng vocab['<PAD>'] để padding

text_indices = [to_indices(t, max_len_text) for t in tokenized_texts]

# --- Dataset và DataLoader ---
class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        # Chuyển đổi danh sách indices và labels thành tensor của PyTorch
        self.texts = torch.tensor(texts, dtype=torch.long) # Indices nên là long
        self.labels = torch.tensor(labels, dtype=torch.long) # Labels cho classification cũng nên là long

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# Chia dữ liệu Train/Test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    text_indices, labels, test_size=0.2, random_state=42, stratify=labels # Thêm stratify để giữ tỷ lệ lớp
)

train_dataset = SentimentDataset(train_texts, train_labels)
test_dataset = SentimentDataset(test_texts, test_labels)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Kích thước từ điển (vocab_size): {vocab_size}")
print(f"Số lượng mẫu huấn luyện: {len(train_dataset)}")
print(f"Số lượng mẫu kiểm tra: {len(test_dataset)}")