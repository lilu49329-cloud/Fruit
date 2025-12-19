import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from PIL import Image
from sklearn.neighbors import NearestNeighbors

# =========================================================
# 1️⃣ LOAD DỮ LIỆU ĐẶC TRƯNG CÓ SẴN
# =========================================================

# Đường dẫn động, tự động lấy theo vị trí file script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = os.path.join(BASE_DIR, "results")
FEATURE_FILE = os.path.join(RESULT_PATH, "features.npy")
PATH_FILE = os.path.join(RESULT_PATH, "paths.npy")
LABEL_FILE = os.path.join(RESULT_PATH, "labels.npy")

if not all(os.path.exists(p) for p in [FEATURE_FILE, PATH_FILE, LABEL_FILE]):
    raise FileNotFoundError("❌ Không tìm thấy file đặc trưng hoặc label trong thư mục results!")


features = np.load(FEATURE_FILE)
paths = np.load(PATH_FILE, allow_pickle=True)
labels = np.load(LABEL_FILE, allow_pickle=True)
# Đảm bảo chỉ lấy đúng 10 loại quả
unique_labels = sorted(list(set(labels)))
if len(unique_labels) > 10:
    print(f"[Warning] Có nhiều hơn 10 loại quả trong labels: {unique_labels}")
    mask = np.isin(labels, unique_labels[:10])
    features = features[mask]
    paths = paths[mask]
    labels = labels[mask]
    print(f"[Info] Đã lọc lại còn {len(set(labels))} loại quả.")

print(f"✅ Đã load {len(features)} đặc trưng từ dataset ({len(set(labels))} lớp).")
print("Feature dim =", features.shape[1])

# =========================================================
# 2️⃣ MODEL TRÍCH ĐẶC TRƯNG GIỐNG NHƯ KHI TRAIN
# =========================================================
base = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224,224,3))

def extract_feature(image_path):
    """Trích đặc trưng ảnh mới (chuẩn hoá y hệt dataset)."""
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    arr = preprocess_input(arr)
    feat = base.predict(arr, verbose=0)
    return feat.flatten()

# =========================================================
# 3️⃣ TÌM KIẾM ẢNH TƯƠNG TỰ (CORRELATION DISTANCE)
# =========================================================
def search_image_from_feature(query_feat, query_path=None, top_k=3, skip_index=None):
    """Tìm kiếm dựa trên đặc trưng có sẵn (đã trích)."""
    nn = NearestNeighbors(n_neighbors=top_k+1 if skip_index is not None else top_k, metric="correlation")
    nn.fit(features)
    dists, indices = nn.kneighbors(query_feat.reshape(1, -1))

    # Nếu query nằm trong dataset, bỏ chính nó ra
    if skip_index is not None:
        indices = indices[0][1:]
        dists = dists[0][1:]
    else:
        indices = indices[0]
        dists = dists[0]

    # Hiển thị kết quả
    plt.figure(figsize=(16, 6))
    if query_path:
        plt.subplot(1, top_k + 1, 1)
        plt.imshow(mpimg.imread(query_path))
        plt.title("Ảnh truy vấn", fontsize=14)
        plt.axis("off")

    print("\n🎯 3 ảnh tương tự nhất:")
    for i, (idx, dist) in enumerate(zip(indices[:top_k], dists[:top_k])):
        img_path = paths[idx]
        print(f"  #{i+1}: {os.path.basename(img_path)} | Label = {labels[idx]} | Dist = {dist:.4f}")
        plt.subplot(1, top_k + 1, i + 2)
        plt.imshow(mpimg.imread(img_path))
        plt.title(f"Tương tự #{i+1}\n{os.path.basename(img_path)}\nDist={dist:.3f}", fontsize=10)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    return indices, dists

# =========================================================
# 4️⃣ KIỂM THỬ ẢNH NGOÀI DATASET
# =========================================================
def test_external_image(query_path, top_k=3):
    print(f"\n🔍 Ảnh ngoài dataset: {os.path.basename(query_path)}")
    if not os.path.exists(query_path):
        print("❌ Không tìm thấy ảnh!")
        return
    query_feat = extract_feature(query_path)
    search_image_from_feature(query_feat, query_path=query_path, top_k=top_k)

# =========================================================
# 5️⃣ KIỂM THỬ ẢNH TRONG DATASET (CHỌN THỦ CÔNG)
# =========================================================
def test_internal_image_manual(top_k=3):
    """Cho phép chọn ảnh trong dataset bằng index hoặc đường dẫn."""
    print("\n📂 Bạn có thể nhập index hoặc đường dẫn của ảnh trong dataset.")
    print(f"Tổng số ảnh trong dataset: {len(paths)}")

    user_input = input("🔹 Nhập chỉ số (0–{}) hoặc đường dẫn: ".format(len(paths)-1)).strip()

    # Người dùng nhập index
    if user_input.isdigit():
        idx = int(user_input)
        if idx < 0 or idx >= len(paths):
            print("❌ Index không hợp lệ!")
            return
        query_path = paths[idx]
        query_label = labels[idx]
        query_feat = features[idx]
        print(f"🖼️ Ảnh chọn: {os.path.basename(query_path)} | Label = {query_label}")
        # KHÔNG loại trừ ảnh truy vấn trong kết quả (trả về cả chính nó nếu là gần nhất)
        neighbor_indices, _ = search_image_from_feature(query_feat, query_path=query_path, top_k=top_k)
        neighbor_labels = labels[neighbor_indices]
        match = np.sum(neighbor_labels == query_label)
        print(f"\n📊 Kết quả: {match}/{top_k} ảnh trùng label.")

    # Người dùng nhập đường dẫn ảnh
    elif os.path.exists(user_input):
        query_path = user_input
        query_feat = extract_feature(query_path)
        search_image_from_feature(query_feat, query_path=query_path, top_k=top_k)

    else:
        print("⚠️ Đầu vào không hợp lệ!")

# =========================================================
# 6️⃣ CHẠY CHƯƠNG TRÌNH
# =========================================================
if __name__ == "__main__":
    # Tự động train và lưu model KNN (correlation) cho backend sử dụng
    from sklearn.neighbors import KNeighborsClassifier
    import joblib
    knn = KNeighborsClassifier(n_neighbors=1, metric="correlation")
    knn.fit(features, labels)
    model_path = os.path.join(RESULT_PATH, "knn_correlation_model.joblib")
    joblib.dump(knn, model_path)
    print(f"\n✅ Đã train và lưu model KNN correlation vào {model_path}")
