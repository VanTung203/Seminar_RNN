import torch
import torch.nn as nn
import numpy as np # Cần cho việc load GloVe

# --- Hàm tiện ích để load GloVe (cần có file GloVe) ---
def load_glove_embeddings(glove_file_path, word_to_idx, embedding_dim):
    """
    Tải GloVe embeddings và tạo ma trận trọng số.
    Args:
        glove_file_path (str): Đường dẫn đến file GloVe (ví dụ: 'glove.6B.100d.txt').
        word_to_idx (dict): Từ điển ánh xạ từ sang index (vocab).
        embedding_dim (int): Kích thước của GloVe embedding (ví dụ: 100).
    Returns:
        torch.Tensor: Ma trận trọng số embedding.
    """
    print(f"Đang tải GloVe embeddings từ: {glove_file_path}")
    embeddings_index = {}
    try:
        with open(glove_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file GloVe tại '{glove_file_path}'. Vui lòng tải và đặt file vào đúng đường dẫn.")
        print("Mô hình sẽ sử dụng embedding học từ đầu (scratch) cho trường hợp 'pretrained=True'.")
        return None
    except Exception as e:
        print(f"Lỗi khi đọc file GloVe: {e}")
        return None


    vocab_size = len(word_to_idx)
    # Khởi tạo ma trận embedding với zero (hoặc ngẫu nhiên nhỏ)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    # embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, embedding_dim)) # Khởi tạo ngẫu nhiên

    loaded_vectors = 0
    for word, i in word_to_idx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Những từ có trong GloVe và trong vocab
            embedding_matrix[i] = embedding_vector
            loaded_vectors += 1
        # Những từ không có trong GloVe (bao gồm <UNK>) sẽ là vector zero (hoặc ngẫu nhiên)
        # <PAD> cũng sẽ là vector zero theo mặc định (nếu index của nó là 0)

    print(f"Tìm thấy {loaded_vectors}/{vocab_size} vector từ trong GloVe.")
    return torch.tensor(embedding_matrix, dtype=torch.float)


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 pretrained=False, vocab_for_glove=None, glove_path=None, freeze_embedding=True):
        super().__init__()
        self.embedding_dim = embedding_dim # Lưu lại để dùng nếu GloVe không load được

        # --- Khởi tạo embedding layer ---
        if pretrained and vocab_for_glove is not None and glove_path is not None:
            # Sử dụng GloVe nếu pretrained=True và có thông tin vocab + đường dẫn GloVe
            weights_matrix = load_glove_embeddings(glove_path, vocab_for_glove, embedding_dim)
            if weights_matrix is not None:
                # num_embeddings phải bằng vocab_size, embedding_dim phải khớp với GloVe
                self.embedding = nn.Embedding.from_pretrained(weights_matrix,
                                                              freeze=freeze_embedding,
                                                              padding_idx=vocab_for_glove.get('<PAD>', 0))
                print("Đã khởi tạo Embedding layer với GloVe.")
            else:
                # Nếu không load được GloVe, quay về học từ đầu
                print("Không thể tải GloVe, Embedding layer sẽ được học từ đầu (scratch).")
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab_for_glove.get('<PAD>', 0))
        else:
            # Học embedding từ đầu (Scratch)
            # padding_idx=0 nếu <PAD> có index là 0 trong vocab
            default_padding_idx = 0
            if vocab_for_glove: # Ưu tiên lấy từ vocab nếu có
                default_padding_idx = vocab_for_glove.get('<PAD>', 0)
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=default_padding_idx)
            if pretrained: # Nếu cờ pretrained=True nhưng không đủ thông tin để load GloVe
                 print("Cờ pretrained=True nhưng không có vocab_for_glove hoặc glove_path. Embedding layer sẽ được học từ đầu (scratch).")
            else:
                print("Đã khởi tạo Embedding layer để học từ đầu (scratch).")


        # --- Khởi tạo khối RNN layer ---
        # Yêu cầu: dùng nn.RNN với 128 hidden units, batch_first=True
        # Input cho RNN là output của Embedding (embedding_dim)
        self.rnn = nn.RNN(input_size=embedding_dim,
                          hidden_size=hidden_dim, # 128 hidden units
                          num_layers=1,           # Đề bài không nói rõ số lớp, mặc định 1
                          batch_first=True,       # Input/output có batch ở chiều đầu tiên
                          bidirectional=False)    # Đề bài không yêu cầu bidirectional

        # --- Khởi tạo tầng Dense để dự đoán 3 nhãn ---
        # Input cho Dense là hidden state cuối cùng từ RNN (hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim) # output_dim là 3

    def forward(self, text):
        # text shape: (batch_size, seq_len)

        # 1. Chuyển text thành embedding
        # embedded shape: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(text)

        # 2. Đưa qua khối RNN để lấy hidden state cuối
        # rnn_output shape (batch_first=True): (batch_size, seq_len, hidden_dim)
        # hidden_last shape: (num_layers * num_directions, batch_size, hidden_dim)
        # Với nn.RNN, num_layers=1, num_directions=1 -> (1, batch_size, hidden_dim)
        rnn_output, hidden_last = self.rnn(embedded)

        # Lấy hidden state của bước thời gian cuối cùng từ lớp RNN cuối cùng.
        # Vì hidden_last có shape (1, batch_size, hidden_dim), ta cần squeeze chiều đầu tiên.
        # Hoặc có thể lấy output của RNN tại step cuối cùng: rnn_output[:, -1, :]
        # Tuy nhiên, với padding, hidden_last thường được ưu tiên hơn vì nó chứa thông tin tổng hợp
        # của chuỗi thực sự (trước khi bị ảnh hưởng bởi padding nếu không dùng packed sequence).
        # Trong trường hợp đơn giản này và không dùng packed sequence, lấy hidden_last là đủ.
        final_hidden_state = hidden_last.squeeze(0) # Shape: (batch_size, hidden_dim)

        # 3. Đưa hidden state qua tầng Dense để dự đoán 3 nhãn
        # predictions shape: (batch_size, output_dim)
        predictions = self.fc(final_hidden_state)

        return predictions # Trả về logits

# Phần khởi tạo mô hình ví dụ ở cuối file model.py gốc không cần thiết nếu
# việc khởi tạo được thực hiện trong train_eval.py
# Tuy nhiên, để chạy thử file này độc lập, ta có thể làm như sau:
if __name__ == '__main__':
    # Các giá trị này cần được lấy từ file data.py hoặc định nghĩa trước
    # Giả sử vocab từ data.py đã được import hoặc tính toán ở đây
    # Ví dụ (cần khớp với data.py):
    example_vocab = {'<PAD>': 0, '<UNK>': 1, 'tôi': 2, 'đi':3, 'làm':4}
    vocab_size_example = len(example_vocab)
    embedding_dim_example = 100 # Theo đề bài
    hidden_dim_example = 128    # Theo đề bài
    output_dim_example = 3      # Theo đề bài

    print("--- Thử nghiệm Scratch Model ---")
    model_scratch = RNNModel(vocab_size=vocab_size_example,
                             embedding_dim=embedding_dim_example,
                             hidden_dim=hidden_dim_example,
                             output_dim=output_dim_example,
                             pretrained=False)
    print(model_scratch)

    print("\n--- Thử nghiệm Pretrained Model (cần file GloVe) ---")
    # path
    glove_file_path_example = 'glove.6B.100d.txt'
    model_pretrained = RNNModel(vocab_size=vocab_size_example,
                                embedding_dim=embedding_dim_example, # Phải khớp với GloVe dim
                                hidden_dim=hidden_dim_example,
                                output_dim=output_dim_example,
                                pretrained=True,
                                vocab_for_glove=example_vocab,
                                glove_path=glove_file_path_example)
    print(model_pretrained)

    # Test thử với một batch giả định
    batch_size_test = 4
    max_len_test = 10 # Giả sử max_len là 10 cho ví dụ này
    dummy_input = torch.randint(0, vocab_size_example, (batch_size_test, max_len_test))

    with torch.no_grad():
        output_logits_scratch = model_scratch(dummy_input)
        output_logits_pretrained = model_pretrained(dummy_input)

    print("\nOutput logits shape (Scratch):", output_logits_scratch.shape)
    print("Output logits shape (Pretrained):", output_logits_pretrained.shape)