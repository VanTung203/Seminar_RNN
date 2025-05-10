import torch
import torch.nn as nn
import numpy as np # Cần cho việc load GloVe

# --- Hàm tiện ích để load GloVe (giữ nguyên) ---
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
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    loaded_vectors = 0
    for word, i in word_to_idx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            loaded_vectors += 1
    print(f"Tìm thấy {loaded_vectors}/{vocab_size} vector từ trong GloVe.")
    return torch.tensor(embedding_matrix, dtype=torch.float)


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 pretrained=False, vocab_for_glove=None, glove_path=None,
                 freeze_embedding=True, is_bidirectional=False, num_rnn_layers=1): # Thêm is_bidirectional và num_rnn_layers
        super().__init__()
        self.embedding_dim = embedding_dim
        self.is_bidirectional = is_bidirectional
        self.num_rnn_layers = num_rnn_layers # Lưu số lớp RNN

        # --- Khởi tạo embedding layer ---
        if pretrained and vocab_for_glove is not None and glove_path is not None:
            weights_matrix = load_glove_embeddings(glove_path, vocab_for_glove, embedding_dim)
            if weights_matrix is not None:
                self.embedding = nn.Embedding.from_pretrained(weights_matrix,
                                                              freeze=freeze_embedding,
                                                              padding_idx=vocab_for_glove.get('<PAD>', 0))
                print("Đã khởi tạo Embedding layer với GloVe.")
            else:
                print("Không thể tải GloVe, Embedding layer sẽ được học từ đầu (scratch).")
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab_for_glove.get('<PAD>', 0))
        else:
            default_padding_idx = 0
            if vocab_for_glove:
                default_padding_idx = vocab_for_glove.get('<PAD>', 0)
            elif vocab_size > 0 and '<PAD>' in getattr(self, 'vocab', {}): # Fallback nếu vocab được truyền vào model
                 default_padding_idx = self.vocab.get('<PAD>', 0)

            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=default_padding_idx)
            if pretrained:
                 print("Cờ pretrained=True nhưng không có vocab_for_glove hoặc glove_path. Embedding layer sẽ được học từ đầu (scratch).")
            else:
                print("Đã khởi tạo Embedding layer để học từ đầu (scratch).")

        # --- Khởi tạo khối RNN layer ---
        # Yêu cầu: dùng nn.RNN với 128 hidden units, batch_first=True
        self.rnn = nn.RNN(input_size=embedding_dim,
                          hidden_size=hidden_dim, # 128 hidden units cho mỗi chiều nếu bidirectional
                          num_layers=self.num_rnn_layers, # Sử dụng số lớp RNN đã lưu
                          batch_first=True,
                          bidirectional=self.is_bidirectional) # Sử dụng cờ is_bidirectional

        # --- Khởi tạo tầng Dense để dự đoán 3 nhãn ---
        # Input cho Dense là hidden state cuối cùng từ RNN
        # Nếu bidirectional, hidden_dim được nhân đôi
        fc_input_dim = hidden_dim * 2 if self.is_bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)

    def forward(self, text):
        # text shape: (batch_size, seq_len)
        embedded = self.embedding(text)
        # embedded shape: (batch_size, seq_len, embedding_dim)

        # Đưa qua khối RNN
        # rnn_output shape (batch_first=True): (batch_size, seq_len, hidden_dim * num_directions)
        # hidden_last shape: (num_layers * num_directions, batch_size, hidden_dim)
        rnn_output, hidden_last = self.rnn(embedded)

        if self.is_bidirectional:
            # hidden_last shape: (num_layers * 2, batch_size, hidden_dim)
            # Ghép hidden state cuối cùng của lớp RNN cuối cùng từ chiều xuôi và chiều ngược.
            # hidden_last[-2,:,:] là fwd hidden của lớp cuối
            # hidden_last[-1,:,:] là bwd hidden của lớp cuối
            # Nếu num_layers = 1:
            # hidden_last[0,:,:] là fwd hidden state cuối cùng
            # hidden_last[1,:,:] là bwd hidden state cuối cùng (tương ứng với token đầu tiên của chuỗi ngược)
            if self.num_rnn_layers == 1:
                final_hidden_state = torch.cat((hidden_last[0, :, :], hidden_last[1, :, :]), dim=1)
            else: # Cho trường hợp nhiều lớp RNN
                 # Lấy hidden state của lớp cuối cùng, chiều xuôi và chiều ngược
                fwd_final = hidden_last[-2,:,:] # Lớp cuối, chiều xuôi
                bwd_final = hidden_last[-1,:,:] # Lớp cuối, chiều ngược
                final_hidden_state = torch.cat((fwd_final, bwd_final), dim=1)

        else: # Unidirectional
            # hidden_last shape: (num_layers * 1, batch_size, hidden_dim)
            # Lấy hidden state của lớp cuối cùng
            final_hidden_state = hidden_last[-1,:,:] # Tương đương hidden_last.squeeze(0) nếu num_layers=1

        # final_hidden_state shape: (batch_size, hidden_dim * num_directions)
        predictions = self.fc(final_hidden_state)
        # predictions shape: (batch_size, output_dim)
        return predictions

# Ví dụ chạy thử (không thay đổi nhiều, chỉ thêm is_bidirectional)
if __name__ == '__main__':
    example_vocab = {'<PAD>': 0, '<UNK>': 1, 'tôi': 2, 'đi':3, 'làm':4, 'học':5, 'vui':6}
    vocab_size_example = len(example_vocab)
    embedding_dim_example = 100
    hidden_dim_example = 128
    output_dim_example = 3
    glove_file_path_example = 'glove.6B.100d.txt' # Đảm bảo file này tồn tại

    print("--- Thử nghiệm Scratch Model (Unidirectional) ---")
    model_scratch_uni = RNNModel(vocab_size=vocab_size_example, embedding_dim=embedding_dim_example,
                                 hidden_dim=hidden_dim_example, output_dim=output_dim_example,
                                 pretrained=False, is_bidirectional=False, num_rnn_layers=1)
    print(model_scratch_uni)

    print("\n--- Thử nghiệm Scratch Model (Bidirectional) ---")
    model_scratch_bi = RNNModel(vocab_size=vocab_size_example, embedding_dim=embedding_dim_example,
                                hidden_dim=hidden_dim_example, output_dim=output_dim_example,
                                pretrained=False, is_bidirectional=True, num_rnn_layers=1)
    print(model_scratch_bi)

    print("\n--- Thử nghiệm Pretrained Model (Unidirectional, cần file GloVe) ---")
    model_pretrained_uni = RNNModel(vocab_size=vocab_size_example, embedding_dim=embedding_dim_example,
                                    hidden_dim=hidden_dim_example, output_dim=output_dim_example,
                                    pretrained=True, vocab_for_glove=example_vocab,
                                    glove_path=glove_file_path_example, is_bidirectional=False, num_rnn_layers=1)
    print(model_pretrained_uni)

    print("\n--- Thử nghiệm Pretrained Model (Bidirectional, cần file GloVe) ---")
    model_pretrained_bi = RNNModel(vocab_size=vocab_size_example, embedding_dim=embedding_dim_example,
                                   hidden_dim=hidden_dim_example, output_dim=output_dim_example,
                                   pretrained=True, vocab_for_glove=example_vocab,
                                   glove_path=glove_file_path_example, is_bidirectional=True, num_rnn_layers=1)
    print(model_pretrained_bi)

    batch_size_test = 4
    max_len_test = 10
    dummy_input = torch.randint(0, vocab_size_example, (batch_size_test, max_len_test))

    with torch.no_grad():
        output_uni = model_scratch_uni(dummy_input)
        output_bi = model_scratch_bi(dummy_input)
    print("\nOutput logits shape (Scratch Uni):", output_uni.shape) # (4, 3)
    print("Output logits shape (Scratch Bi):", output_bi.shape)   # (4, 3)