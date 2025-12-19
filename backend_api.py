
import os
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.neighbors import NearestNeighbors
import joblib
from PIL import Image
import io
import sqlite3

# ==== Paths/config ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(RESULT_PATH, "knn_correlation_model.joblib")
FEATURE_FILE = os.path.join(RESULT_PATH, "features.npy")
PATH_FILE = os.path.join(RESULT_PATH, "paths.npy")
LABEL_FILE = os.path.join(RESULT_PATH, "labels.npy")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")


# ==== App init and CORS ====
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Dev: allow all for frontend testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== SQLite DB for query history ====
DB_PATH = os.path.join(BASE_DIR, "query_history.db")
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            predicted_label TEXT,
            top_similar TEXT
        )
    """)
    conn.commit()
    conn.close()
init_db()

def save_history(filename, predicted_label, top_similar):
    import datetime, json
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO history (timestamp, filename, predicted_label, top_similar) VALUES (?, ?, ?, ?)",
              (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), filename, predicted_label, json.dumps(top_similar)))
    conn.commit()
    conn.close()

# ==== Model/resource loading (on startup) ====
model = None
base_model = None
features = None
paths = None
labels = None
all_classes = None
nn = None

def load_resources():
    global model, base_model, features, paths, labels, all_classes, nn
    model = joblib.load(MODEL_PATH)
    base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
    features = np.load(FEATURE_FILE)
    paths = np.load(PATH_FILE, allow_pickle=True)
    labels = np.load(LABEL_FILE, allow_pickle=True)
    # Đảm bảo chỉ lấy đúng 10 loại quả, loại bỏ các nhãn ngoài (nếu có)
    unique_labels = sorted(list(set(labels)))
    if len(unique_labels) > 10:
        print(f"[Warning] Có nhiều hơn 10 loại quả trong labels: {unique_labels}")
        # Lấy đúng 10 loại đầu tiên (theo thứ tự alphabet)
        mask = np.isin(labels, unique_labels[:10])
        features = features[mask]
        paths = paths[mask]
        labels = labels[mask]
        print(f"[Info] Đã lọc lại còn {len(set(labels))} loại quả.")
    all_classes = sorted(list(set(labels)))
    nn = NearestNeighbors(n_neighbors=3, metric="correlation")
    nn.fit(features)
load_resources()

# ==== Utils ====
def extract_feature(pil_image):
    img_resized = pil_image.resize((224, 224))
    arr = np.expand_dims(np.array(img_resized, dtype=np.float32), axis=0)
    arr = preprocess_input(arr)
    feat = base_model.predict(arr, verbose=0)
    return feat

def get_relative_dataset_path(abspath):
    # Remove BASE_DIR and leading slashes for serving
    rel = os.path.relpath(abspath, BASE_DIR)
    return rel.replace("\\", "/")

# ==== API: Predict ====

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        feat = extract_feature(pil_img)
        pred = model.predict(feat)[0]

        dists, indices = nn.kneighbors(feat)
        top_list = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            dist = float(dists[0][i])
            img_path = paths[idx]
            rel_img_path = get_relative_dataset_path(img_path)
            top_list.append({"path": rel_img_path, "distance": dist})

        # Lưu lịch sử vào database
        save_history(file.filename, str(pred), top_list)

        return JSONResponse({
            "predicted_label": str(pred),
            "top_similar": top_list,
            "all_classes": [str(c) for c in all_classes]
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ==== Static Image Serving (preview similar) ====
@app.get("/dataset/{img_path:path}")
def get_dataset_image(img_path: str):
    abs_img_path = os.path.abspath(os.path.join(DATASET_DIR, img_path))
    if not os.path.exists(abs_img_path):
        return JSONResponse({"error": "Image not found"}, status_code=404)
    return FileResponse(abs_img_path)


# ==== API: Lấy lịch sử truy vấn ====
@app.get("/history")
def get_history():
    import sqlite3, json
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT timestamp, filename, predicted_label, top_similar FROM history ORDER BY id DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()
    history = []
    for row in rows:
        timestamp, filename, predicted_label, top_similar = row
        try:
            top_similar = json.loads(top_similar)
        except:
            top_similar = []
        history.append({
            "timestamp": timestamp,
            "filename": filename,
            "predicted_label": predicted_label,
            "top_similar": top_similar
        })
    return {"history": history}

# ==== Healthcheck ====
@app.get("/")
def index():
    return {"status": "ok"}
