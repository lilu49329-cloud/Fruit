import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.neighbors import NearestNeighbors

import os
import datetime
RESULT_PATH = os.path.join(os.path.dirname(__file__), "results")
MODEL_PATH = os.path.join(RESULT_PATH, "knn_correlation_model.joblib")
FEATURE_FILE = os.path.join(RESULT_PATH, "features.npy")
PATH_FILE = os.path.join(RESULT_PATH, "paths.npy")
LABEL_FILE = os.path.join(RESULT_PATH, "labels.npy")

# Load model & data before any menu logic to ensure base_model is always defined
@st.cache_resource
def load_resources():
    model = joblib.load(MODEL_PATH)
    base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
    features = np.load(FEATURE_FILE)
    paths = np.load(PATH_FILE, allow_pickle=True)
    labels = np.load(LABEL_FILE, allow_pickle=True)
    return model, base_model, features, paths, labels

try:
    model, base_model, features, paths, labels = load_resources()
    classes = sorted(list(set(labels)))
except Exception as err:
    st.error(f"❌ Lỗi khi tải model/data/feature: {err}")
    st.stop()

# Sử dụng st.radio cho sidebar menu để đảm bảo chọn tab hoạt động tốt
menu_options = [
    "🏠 Trang chủ",
    "🍎 Nhận diện quả",
    "🔍 Tìm kiếm",
    "🕑 Lịch sử",
    "📖 Hướng dẫn sử dụng",
    "📈 Báo cáo"
]

# Sidebar menu with st.radio (clean, no JS)
st.sidebar.markdown("""
<style>
.sidebar-menu-label {
    font-size: 1.45rem;
    color: #229954;
    font-weight: bold;
    letter-spacing: 1px;
    margin-bottom: 0;
    text-align: center;
    padding: 14px 0 6px 0;
    border-radius: 18px 18px 0 0;
    background: linear-gradient(90deg,#e0ffe5 0%,#fffbe5 100%);
    box-shadow: 0 2px 8px rgba(40,116,166,0.10);
}
.sidebar-radio {
    background: linear-gradient(180deg,#e0ffe5 0%,#fffbe5 100%);
    border-radius: 0 0 18px 18px;
    padding: 0 0 0 0;
    box-shadow: 0 2px 12px rgba(40,116,166,0.10);
    margin-bottom: 0;
}
.sidebar-radio label {
    font-size: 1.22rem !important;
    font-weight: 700 !important;
    color: #229954 !important;
    padding: 14px 0 14px 0 !important;
    border-radius: 12px !important;
    margin-bottom: 2px !important;
    margin-top: 0 !important;
    transition: background 0.08s, color 0.08s;
    display: flex;
    align-items: center;
    gap: 12px;
}
.sidebar-radio label:hover {
    background: linear-gradient(90deg,#ffecd2 0%,#fcb69f 100%) !important;
    color: #ff5e62 !important;
}
.sidebar-radio input[type="radio"]:checked + label {
    background: linear-gradient(90deg,#43ea5e 0%,#ffb347 100%) !important;
    color: #fff !important;
    box-shadow: 0 2px 8px rgba(255,94,98,0.12);
}
.sidebar-radio {
    margin-bottom: 0 !important;
}
.sidebar-radio:last-child {
    margin-bottom: 0 !important;
}
</style>
<div class='sidebar-menu-label'>Menu</div>
<div class='sidebar-radio'>
""", unsafe_allow_html=True)
if "menu_selected" not in st.session_state:
    st.session_state.menu_selected = menu_options[0]
menu = st.sidebar.radio("", menu_options, index=menu_options.index(st.session_state.menu_selected))
st.sidebar.markdown("</div>", unsafe_allow_html=True)
st.session_state.menu_selected = menu

if menu == "🏠 Trang chủ":
    st.markdown("""
    <style>
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-18px); }
    }
    .fruit-anim {
        display: inline-block;
        animation: bounce 1.2s infinite;
        margin: 0 12px;
    }
    .welcome-anim {
        animation: fadeIn 1.2s;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .center-btn {
        display: flex;
        justify-content: center;
        margin-top: 32px;
    }
    .cta-btn {
        background: linear-gradient(90deg,#ff9800,#ff5e62 60%,#229954 100%);
        color: #fff;
        font-size: 1.35rem;
        font-weight: 700;
        border: none;
        border-radius: 16px;
        padding: 18px 54px;
        cursor: pointer;
        box-shadow: 0 8px 32px rgba(255,94,98,0.18), 0 2px 8px rgba(40,116,166,0.18);
        transition: background 0.3s, transform 0.18s, box-shadow 0.18s;
        outline: none;
        letter-spacing: 0.5px;
        animation: pulse 1.2s infinite;
    }
    .cta-btn:hover {
        background: linear-gradient(90deg,#229954,#ff9800 60%,#ff5e62 100%);
        transform: scale(1.07);
        box-shadow: 0 12px 40px rgba(255,94,98,0.22), 0 4px 16px rgba(40,116,166,0.22);
    }
    @keyframes pulse {
        0% { box-shadow: 0 8px 32px rgba(255,94,98,0.18), 0 2px 8px rgba(40,116,166,0.18); }
        50% { box-shadow: 0 16px 48px rgba(255,94,98,0.28), 0 6px 24px rgba(40,116,166,0.28); }
        100% { box-shadow: 0 8px 32px rgba(255,94,98,0.18), 0 2px 8px rgba(40,116,166,0.18); }
    }
    </style>
    <div class='welcome-anim' style='padding:32px 0; text-align:center;'>
        <span class='fruit-anim'><img src='https://cdn-icons-png.flaticon.com/512/415/415733.png' width='64' title='Dâu'></span>
        <span class='fruit-anim'><img src='https://cdn-icons-png.flaticon.com/512/590/590685.png' width='64' title='Táo'></span>
        <span class='fruit-anim'><img src='https://cdn-icons-png.flaticon.com/512/415/415734.png' width='64' title='Cam'></span>
        <span class='fruit-anim'><img src='https://cdn-icons-png.flaticon.com/512/766/766114.png' width='64' title='Dưa hấu'></span>
        <span class='fruit-anim'><img src='https://cdn-icons-png.flaticon.com/512/415/415735.png' width='64' title='Xoài'></span>
        <span class='fruit-anim'><img src='https://cdn-icons-png.flaticon.com/512/135/135620.png' width='64' title='Dứa'></span>
        <h1 style='color:#2874A6; margin-top:24px;'>Chào mừng đến với hệ thống nhận diện trái cây!</h1>
        <p style="font-size:1.2rem; color:#2d2d2d; line-height:1.8;">
            <b style="font-size:1.3rem; color:#1a7f3c;">✨ Khám phá thế giới trái cây thông minh!</b>
            Hệ thống giúp bạn <span style="color:#1a7f3c;">nhận diện hoa quả</span> và 
            <span style="color:#1a7f3c;">tìm kiếm hình ảnh tương tự</span> nhanh chóng, chính xác chỉ với vài thao tác.<br>
            Dễ dàng sử dụng <b>menu bên trái</b> để khám phá các tính năng tiện ích: 
            <b>Nhận diện</b> – <b>Tìm kiếm</b> – <b>Lịch sử</b> – <b>Hướng dẫn</b> – <b>Báo cáo kiểm thử</b>.<br><br>
        </p>
    """, unsafe_allow_html=True)
    # Căn giữa nút ngay dưới đoạn mô tả, không ảnh hưởng toàn web
    st.markdown("""
    <style>
    .btn-center-under-desc {
        width: 100%;
        display: flex;
        justify-content: center;
        margin-top: -18px;
        margin-bottom: 0;
    }
    div[data-testid="stButton"] button {
        background: linear-gradient(90deg,#ffe5b2 0%,#ffb347 50%,#43ea5e 100%) !important;
        color: #fff !important;
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 16px 44px !important;
        cursor: pointer !important;
        box-shadow: 0 4px 16px rgba(255,94,98,0.10), 0 2px 8px rgba(40,116,166,0.10) !important;
        transition: background 0.3s, transform 0.18s, box-shadow 0.18s !important;
        outline: none !important;
        letter-spacing: 0.5px !important;
        animation: pulse 1.2s infinite;
        position: relative;
        z-index: 2;
        text-shadow: none !important;
    }
    div[data-testid="stButton"] button:hover {
        background: linear-gradient(90deg,#43ea5e 0%,#ffb347 60%,#ffe5b2 100%) !important;
        transform: scale(1.04);
        box-shadow: 0 8px 24px rgba(255,94,98,0.14), 0 4px 12px rgba(40,116,166,0.14) !important;
    }
    @keyframes pulse {
        0% { box-shadow: 0 4px 16px rgba(255,94,98,0.10), 0 2px 8px rgba(40,116,166,0.10); }
        50% { box-shadow: 0 8px 24px rgba(255,94,98,0.14), 0 4px 12px rgba(40,116,166,0.14); }
        100% { box-shadow: 0 4px 16px rgba(255,94,98,0.10), 0 2px 8px rgba(40,116,166,0.10); }
    }
    div[data-testid="stButton"] button span.rocket {
        display: inline-block;
        font-size: 1.4rem;
        margin-left: 8px;
        vertical-align: middle;
        animation: rocketmove 1.2s infinite;
    }
    @keyframes rocketmove {
        0% { transform: translateY(0) rotate(-8deg); }
        40% { transform: translateY(-4px) rotate(8deg); }
        60% { transform: translateY(-7px) rotate(-8deg); }
        100% { transform: translateY(0) rotate(-8deg); }
    }
    </style>
    <div class='btn-center-under-desc'>
    """, unsafe_allow_html=True)
    if st.button("Bắt đầu ngay 🚀", key="start_btn"):
        st.session_state.menu_selected = "🍎 Nhận diện quả"
    st.markdown("</div>", unsafe_allow_html=True)

elif menu == "🍎 Nhận diện quả":
    st.markdown("<h2 style='color:#2874A6;'>🍎 NHẬN DIỆN QUẢ</h2>", unsafe_allow_html=True)
    img = None
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], key="upload")
    st.markdown("""
    <style>
    /* Đổi placeholder của file uploader sang tiếng Việt */
    .stFileUploader label {display:none;}
    .stFileUploader div[aria-label="Drag and drop file here"]:before {
        content: "Kéo và thả ảnh tại đây";
        color: #2874A6;
        font-size: 1.1rem;
        position: absolute;
        left: 0; right: 0; top: 0; bottom: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 2;
        background: rgba(255,255,255,0.7);
    }
    </style>
    """, unsafe_allow_html=True)
    if uploaded:
        try:
            img = Image.open(uploaded).convert("RGB")
        except Exception:
            st.error("❌ Ảnh không hợp lệ hoặc bị lỗi. Vui lòng thử lại!")
            img = None
    if img is not None:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(img, caption="Ảnh bạn vừa chọn", width=250)
        # Tạo thư mục lưu truy vấn nếu chưa có
        query_dir = os.path.join(RESULT_PATH, "queries")
        os.makedirs(query_dir, exist_ok=True)
        # Tạo tên file duy nhất theo timestamp
        query_filename = f"query_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        query_filepath = os.path.join(query_dir, query_filename)
        img.save(query_filepath)
        try:
            # Tiền xử lý ảnh đầu vào
            img_resized = img.resize((224, 224))
            img_array = np.array(img_resized)
            img_preproc = preprocess_input(img_array)
            feat = base_model.predict(np.expand_dims(img_preproc, axis=0))
            pred = model.predict(feat)[0]
            st.markdown(f"<div class='result-label'>🎯 Kết quả dự đoán: <span class='result-title'>{pred}</span></div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Lỗi xử lý model nhận diện: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

elif menu == "🔍 Tìm kiếm":
    st.markdown("<h2 style='color:#2874A6;'>🔍 TÌM KIẾM ẢNH TƯƠNG TỰ</h2>", unsafe_allow_html=True)
    img_search = None
    uploaded_search = st.file_uploader("", type=["jpg", "jpeg", "png"], key="search")
    st.markdown("""
    <style>
    .stFileUploader label {display:none;}
    .stFileUploader div[aria-label="Drag and drop file here"]:before {
        content: "Kéo và thả ảnh tại đây";
        color: #2874A6;
        font-size: 1.1rem;
        position: absolute;
        left: 0; right: 0; top: 0; bottom: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 2;
        background: rgba(255,255,255,0.7);
    }
    </style>
    """, unsafe_allow_html=True)
    if uploaded_search:
        try:
            img_search = Image.open(uploaded_search).convert("RGB")
        except Exception:
            st.error("❌ Ảnh không hợp lệ hoặc bị lỗi. Vui lòng thử lại!")
            img_search = None
    if img_search is not None:
        st.image(img_search, caption="Ảnh bạn vừa chọn", width=250)
        # Tạo thư mục lưu truy vấn nếu chưa có
        query_dir = os.path.join(RESULT_PATH, "queries")
        os.makedirs(query_dir, exist_ok=True)
        # Tạo tên file duy nhất theo timestamp
        query_filename = f"query_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        query_filepath = os.path.join(query_dir, query_filename)
        img_search.save(query_filepath)
        try:
            # Trích xuất đặc trưng từ ảnh truy vấn
            img_resized = img_search.resize((224, 224))
            img_array = np.array(img_resized)
            img_preproc = preprocess_input(img_array)
            query_feat = base_model.predict(np.expand_dims(img_preproc, axis=0))

            # Dùng KNN tìm 3 ảnh tương tự nhất
            k = min(3, len(features))
            knn = NearestNeighbors(n_neighbors=k, metric="cosine")
            knn.fit(features)
            dists, idxs = knn.kneighbors(query_feat)
            dists = dists[0]
            idxs = idxs[0]
            # Loại bỏ ảnh trùng chính ảnh truy vấn 
            unique_img_idxs = []
            unique_dists = []
            img_query_feat = query_feat.flatten()
            for img_idx, dist in zip(idxs, dists):
                candidate_feat = features[img_idx].flatten()
                if not np.allclose(candidate_feat, img_query_feat):  # Loại bỏ nếu đặc trưng trùng khớp ảnh query
                    unique_img_idxs.append(img_idx)
                    unique_dists.append(dist)
                if len(unique_img_idxs) >= 3:
                    break
            st.markdown("<h4 style='color:#2874A6;'>🔍 3 ảnh tương tự nhất trong dataset</h4>", unsafe_allow_html=True)
            cols = st.columns(4)
            with cols[0]:
                st.markdown("""
                    <div style='text-align:center; padding:10px;'>
                        <span style='font-weight:bold; color:#2874A6;'>Ảnh truy vấn</span>
                    </div>
                """, unsafe_allow_html=True)
                st.image(img_search, caption=None)
            for i, (img_idx, dist) in enumerate(zip(unique_img_idxs, unique_dists), start=1):
                sim_img_rel = paths[img_idx]
                sim_img_path = os.path.join(os.path.dirname(RESULT_PATH), sim_img_rel)
                if os.path.exists(sim_img_path):
                    with cols[i]:
                        st.markdown(f"""
                            <div style='text-align:center; padding:10px;'>
                                <span style='font-weight:bold; color:#229954;'>Tương tự #{i}</span><br>
                                <span style='color:#888;'>Corr={1-dist:.3f}</span>
                            </div>
                        """, unsafe_allow_html=True)
                        st.image(sim_img_path, caption=None)
        except Exception as e:
            st.error(f"❌ Lỗi xử lý tìm kiếm tương tự: {e}")

elif menu == "🕑 Lịch sử":
    st.markdown("<div class='card'><h2 style='color:#2874A6;'>🕑 Lịch sử truy vấn gần đây</h2></div>", unsafe_allow_html=True)
    # Lưu truy vấn lịch sử tạm thời trong session_state
    if "history_queries" not in st.session_state:
        st.session_state.history_queries = []

    # Merge các truy vấn từ thư mục (nếu có) và session_state cho hiển thị demo
    query_dir = os.path.join(RESULT_PATH, "queries")
    history_files = []
    if os.path.exists(query_dir):
        history_files = sorted(os.listdir(query_dir), reverse=True)[:10]

    if not history_files and not st.session_state.history_queries:
        st.info("Chưa có lịch sử truy vấn nào.")
    else:
        for filename in history_files:
            query_img_path = os.path.join(query_dir, filename)
            if os.path.exists(query_img_path):
                st.markdown(f"<div class='history-item'><b>Tên file:</b> {filename}</div>", unsafe_allow_html=True)
                st.image(query_img_path, caption="Ảnh truy vấn", width=150)
                # Hiển thị thêm 3 ảnh tương tự nhất bằng KNN
                try:
                    img_hist = Image.open(query_img_path).convert("RGB")
                    img_resized = img_hist.resize((224, 224))
                    img_array = np.array(img_resized)
                    img_preproc = preprocess_input(img_array)
                    hist_feat = base_model.predict(np.expand_dims(img_preproc, axis=0))
                    # Dùng KNN (corr) lấy 3 ảnh tương tự nhất trong dataset (loại trừ trùng đặc trưng query)
                    k_hist = min(3, len(features))
                    knn = NearestNeighbors(n_neighbors=k_hist, metric="cosine")
                    knn.fit(features)
                    dists, idxs = knn.kneighbors(hist_feat)
                    dists = dists[0]
                    idxs = idxs[0]
                    unique_img_idxs = []
                    unique_dists = []
                    img_query_feat = hist_feat.flatten()
                    for img_idx, dist in zip(idxs, dists):
                        candidate_feat = features[img_idx].flatten()
                        if not np.allclose(candidate_feat, img_query_feat):
                            unique_img_idxs.append(img_idx)
                            unique_dists.append(dist)
                        if len(unique_img_idxs) >= 3:
                            break
                    cols = st.columns(4)
                    with cols[0]:
                        st.markdown(
                            "<div style='text-align:center;padding:10px;'><span style='font-weight:bold; color:#2874A6;'>Ảnh truy vấn</span></div>",
                            unsafe_allow_html=True)
                        st.image(query_img_path, caption=None)
                    for i, (img_idx, dist) in enumerate(zip(unique_img_idxs, unique_dists), start=1):
                        sim_img_rel = paths[img_idx]
                        sim_img_path = os.path.join(os.path.dirname(RESULT_PATH), sim_img_rel)
                        if os.path.exists(sim_img_path):
                            with cols[i]:
                                st.markdown(f"""
                                    <div style='text-align:center; padding:10px;'>
                                        <span style='font-weight:bold; color:#229954;'>Tương tự #{i}</span><br>
                                        <span style='color:#888;'>Corr={1-dist:.3f}</span>
                                    </div>
                                """, unsafe_allow_html=True)
                                st.image(sim_img_path, caption=None)
                except Exception as e:
                    st.error(f"Lỗi tìm kiếm tương tự lịch sử: {e}")
    # Có thể mở rộng hiển thị thêm thông tin nếu muốn lưu thêm metadata


elif menu == "📖 Hướng dẫn sử dụng":
    st.markdown("<h2 style='color:#2874A6;'>📖 Hướng dẫn sử dụng</h2>", unsafe_allow_html=True)
    st.markdown("""
    **Bước 1:** Chọn chế độ truy vấn ảnh (tải lên hoặc chọn từ dataset).
    
    **Bước 2:** Tải lên ảnh quả bất kỳ hoặc chọn ảnh có sẵn.
    
    **Bước 3:** Xem kết quả nhận diện và các ảnh tương tự nhất trong dataset.
    
    **Lưu ý:**
    - Ảnh tải lên nên rõ nét, có quả nằm giữa ảnh.
    - Hệ thống sẽ dự đoán loại quả và hiển thị 3 ảnh giống nhất.
    
    ---
    ### 📊 Kiểm thử độ chính xác model
    - Để kiểm tra độ chính xác, bạn có thể chọn ảnh trong dataset.
    - Hệ thống sẽ so sánh label dự đoán với label thực tế.
    - Độ chính xác trên tập huấn luyện thường cao, với ảnh thực tế cần kiểm thử thêm.
    """)

elif menu == "📈 Báo cáo":
    st.markdown("<h2 style='color:#2874A6;'>📈 Báo cáo kiểm thử model AI</h2>", unsafe_allow_html=True)


# Function and call for model test report - placed AFTER model, features, labels have loaded
if menu == "📈 Báo cáo":
    def report_model_accuracy():
        correct = 0
        total = len(features)
        for i in range(total):
            feat = features[i].reshape(1, -1)
            pred = model.predict(feat)[0]
            if pred == labels[i]:
                correct += 1
        st.info(f"**Độ chính xác nhận diện trên dataset: {correct}/{total} ({correct/total:.2%})**")
    report_model_accuracy()



st.markdown(
    """
    <div style='text-align:center; color:#888; font-size:1rem; margin-top:4px;'>
        © 2025 Fruit System | Powered by Nhom 4
    </div>
    """,
    unsafe_allow_html=True
)
