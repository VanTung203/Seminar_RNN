import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
import nltk
from sklearn.model_selection import train_test_split
from collections import Counter
import re # Thêm thư viện re để xử lý biểu thức chính quy (loại bỏ dấu câu)

nltk.download('punkt', quiet=True) # Thêm quiet=True để không in ra mỗi lần chạy nếu đã tải

# --- Danh sách Stop Words tiếng Việt (ví dụ) ---
VIETNAMESE_STOP_WORDS = set("""a à ả ã á ạ ă ằ ẳ ẵ ắ ặ â ầ ẩ ẫ ấ ậ b c d đ e è ẻ ẽ é ẹ ê ề ể ễ ế ệ
    f g h i ì ỉ ĩ í ị j k l m n o ò ỏ õ ó ọ ô ồ ổ ỗ ố ộ ơ ờ ở ỡ ớ ợ
    p q r s t u ù ủ ũ ú ụ ư ừ ử ữ ứ ự v w x y ỳ ỷ ỹ ý ỵ z
    bị bởi cả các cái cần chẳng chứ chưa có có_thể cùng của
    đã đang đấy để đến đều điều đó được gì hơn hoặc là lại
    lên lúc mà mỗi một nếu như những nhưng như ở ra rằng rồi
    sau sẽ thì trong từng và với vì vậy
    ai anh chị em ông bà cô chú bác bạn tôi mình chúng_tôi
    chúng_ta mọi_người người_ta khi nào sao gì đâu bao_giờ bao_lâu
    trước sau trên dưới trong ngoài giữa bên_cạnh bên_trong bên_ngoài
    thường_xuyên luôn luôn thỉnh_thoảng đôi_khi ngay_lập_tức vừa_mới
    rất quá lắm cực_kỳ vô_cùng khá hơi một_chút một_ít
    và hoặc hay là thì mà rằng để cho của với bởi vì tại bởi_vì
    đi đến làm vào ra lên xuống qua lại xem đọc gửi nhận tham_gia
    chuẩn_bị kiểm_tra cập_nhật tìm đợi cài_đặt soạn_thảo đăng_ký
    điền trả_lời sắp_xếp
    hôm_nay ngày_mai hôm_qua bây_giờ lúc_này ngay_bây_giờ sáng_nay chiều_nay
    tuần_này tháng_này quý này năm_nay
    công_việc dự_án bài_tập buổi_họp khóa_học email báo_cáo tài_liệu
    hệ_thống công_ty trường lớp sếp đồng_nghiệp khách_hàng giảng_viên
    nhân_viên quản_lý sinh_viên giáo_viên
    vấn_đề quy_trình kế_hoạch mục_tiêu chỉ_tiêu thông_tin
    kỹ_năng kinh_nghiệm phương_pháp giải_pháp
    lịch lịch_trình chính_sách thông_báo nội_dung yêu_cầu kết_quả
    máy_tính phần_mềm mạng internet thiết_bị
    số_một số_hai con_số này kia đó ấy những_thứ_đó
    cái_gì ai_đó một_số một_vài tất_cả mọi mỗi
    xin_chào tạm_biệt cảm_ơn xin_lỗi làm_ơn có_lẽ chắc_chắn
    ừ vâng dạ không có thể_là được_rồi
    thứ_hai thứ_ba thứ_tư thứ_năm thứ_sáu thứ_bảy chủ_nhật
    tháng_một tháng_hai tháng_ba tháng_tư tháng_năm tháng_sáu tháng_bảy
    tháng_tám tháng_chín tháng_mười tháng_mười_một tháng_mười_hai
    ông bà anh chị em tôi bạn họ chúng_tôi các_bạn
    đang đã sẽ vẫn còn mới chỉ vừa
    diễn_ra đơn_giản chỉ_là
    """.split())

SINGLE_CHAR_WORDS_TO_REMOVE = set([char for char in "abcdefghijklmnopqrstuvwxyzđ"])
VIETNAMESE_STOP_WORDS.update(SINGLE_CHAR_WORDS_TO_REMOVE)

# --- Hàm tiền xử lý văn bản ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    # Thêm len(word) > 1 để loại bỏ từ đơn ký tự sau khi tokenize và trước khi check stopword
    tokens = [word for word in tokens if word not in VIETNAMESE_STOP_WORDS and word.strip() and len(word) > 1]
    return tokens

# --- Đọc dữ liệu từ CSV ---
try:
    data = pd.read_csv('sentiment_data.csv').dropna()
    print(f"Đã tải dữ liệu gốc: {len(data)} mẫu.")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'sentiment_data.csv'. Vui lòng tạo file này.")
    exit()

texts_original = data['text'].tolist() # Đổi tên biến để giữ lại văn bản gốc nếu cần
labels = data['label'].map({'Positive': 0, 'Negative': 1, 'Neutral': 2}).tolist()

# --- ÁP DỤNG TIỀN XỬ LÝ, Tokenize và xây dựng từ điển ---
print("Đang tiền xử lý văn bản...")
# SỬ DỤNG HÀM PREPROCESS_TEXT Ở ĐÂY:
tokenized_texts = [preprocess_text(text) for text in texts_original]

# In ra một vài ví dụ sau khi tiền xử lý để kiểm tra
print("\nVí dụ văn bản sau tiền xử lý:")
for i in range(min(5, len(texts_original))): # Sử dụng texts_original để in bản gốc
    print(f"Gốc: {texts_original[i]}")
    print(f"Sau xử lý: {tokenized_texts[i]}")
    print("-" * 20)

all_words = [w for txt in tokenized_texts for w in txt]

if not all_words:
    print("LỖI: Không có từ nào còn lại sau khi tiền xử lý. Kiểm tra lại danh sách stop words hoặc dữ liệu đầu vào.")
    vocab = {'<PAD>': 0, '<UNK>': 1, 'empty_token_after_preprocess': 2}
    print("Đã tạo từ điển tối thiểu do không có từ nào sau tiền xử lý.")
else:
    num_most_common = min(4998, len(Counter(all_words)))
    most_common = Counter(all_words).most_common(num_most_common)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for i, (w, _) in enumerate(most_common, 2):
        vocab[w] = i

vocab_size = len(vocab)
print(f"\nKích thước từ điển sau tiền xử lý (vocab_size): {vocab_size}")
if vocab_size < 10:
    print("CẢNH BÁO: Kích thước từ điển rất nhỏ. Có thể do tiền xử lý quá mạnh hoặc dữ liệu đầu vào ít từ vựng.")


# --- Hàm chuyển đổi tokens thành indices và padding ---
max_len_text = 50
def to_indices(tokens, max_len):
    idxs = [vocab.get(t, vocab['<UNK>']) for t in tokens][:max_len]
    return idxs + [vocab['<PAD>']] * (max_len - len(idxs))

text_indices = [to_indices(t, max_len_text) for t in tokenized_texts]

# --- Dataset và DataLoader ---
class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

if not labels:
    print("LỖI: Không có nhãn nào được tải. Kiểm tra file CSV và ánh xạ nhãn.")
    train_texts, test_texts, train_labels, test_labels = [], [], [], []
else:
    try:
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            text_indices, labels, test_size=0.2, random_state=42, stratify=labels
        )
    except ValueError as e:
        print(f"Lỗi khi chia train/test (có thể do số lượng mẫu mỗi lớp quá ít sau khi dropna): {e}")
        print("Thử chia không dùng stratify...")
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            text_indices, labels, test_size=0.2, random_state=42
        )

if not train_texts or not test_texts:
    print("LỖI: Dữ liệu huấn luyện hoặc kiểm tra rỗng sau khi chia. Kiểm tra lại dữ liệu và quá trình tiền xử lý.")
    train_dataset = SentimentDataset([], [])
    test_dataset = SentimentDataset([], [])
else:
    train_dataset = SentimentDataset(train_texts, train_labels)
    test_dataset = SentimentDataset(test_texts, test_labels)

batch_size = 32
if len(train_dataset) > 0:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
else:
    print("CẢNH BÁO: Train dataset rỗng, không thể tạo train_loader.")
    train_loader = None

if len(test_dataset) > 0:
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
else:
    print("CẢNH BÁO: Test dataset rỗng, không thể tạo test_loader.")
    test_loader = None

print(f"Số lượng mẫu huấn luyện: {len(train_dataset) if train_dataset else 0}")
print(f"Số lượng mẫu kiểm tra: {len(test_dataset) if test_dataset else 0}")

if train_loader:
    print(f"Số batch trong train_loader: {len(train_loader)}")
if test_loader:
    print(f"Số batch trong test_loader: {len(test_loader)}")

if 'vocab' not in globals(): # Đảm bảo vocab tồn tại
    vocab = {'<PAD>': 0, '<UNK>': 1}
    vocab_size = len(vocab)

# Thêm phần chạy thử nếu file này được chạy trực tiếp
if __name__ == '__main__':
    print("\n--- Chạy thử data.py ---")
    print(f"Kích thước từ điển cuối cùng: {vocab_size}")
    if train_loader and len(train_dataset) > 0:
        print("Lấy một batch từ train_loader:")
        try:
            sample_texts_indices, sample_labels = next(iter(train_loader))
            print("Texts indices shape:", sample_texts_indices.shape)
            print("Labels shape:", sample_labels.shape)
            print("Một vài indices của văn bản đầu tiên trong batch:", sample_texts_indices[0][:20].tolist())
            # In ngược lại một vài token
            idx_to_word = {idx: word for word, idx in vocab.items()}
            print("Token tương ứng:", [idx_to_word.get(idx.item(), '<UNK>') for idx in sample_texts_indices[0][:20]])
        except StopIteration:
            print("Train loader rỗng, không lấy được batch.")
        except Exception as e:
            print(f"Lỗi khi lấy batch từ train_loader: {e}")
    else:
        print("Train loader không được tạo hoặc rỗng.")