# Hướng dẫn deploy backend API (FastAPI) và frontend (Streamlit) lên Render

## I. Cấu trúc deploy tiêu chuẩn

### 1. runtime.txt *(BẮT BUỘC)*
- Đảm bảo file `runtime.txt` với nội dung:
  ```
  python-3.11.8
  ```
- File này bắt buộc với Render để đảm bảo sử dụng đúng Python version, giúp tránh lỗi build scipy/numpy không có wheel trên các version Python mới (>=3.13).

### 2. Backend (FastAPI)
#### a. File cần có:
- `backend_api.py`
- `requirements-backend.txt` (chỉ chứa dependencis backend, ML, API)
- `runtime.txt`
- `Procfile`
- Thư mục dữ liệu/model (`dataset/`, `results/`...)

#### b. requirements-backend.txt ví dụ:
```
pip>=25.3
setuptools>=80.9.0
wheel>=0.40.0
numpy==1.24.3
scipy==1.10.1
scikit-learn==1.3.2
tensorflow==2.13.0
scikit-image==0.23.2
matplotlib==3.7.2
pillow==10.4.0
joblib==1.3.2
fastapi==0.95.2
uvicorn==0.23.2
python-multipart==0.0.9
requests==2.32.3
```

#### c. Procfile ví dụ:
```
web: uvicorn backend_api:app --host 0.0.0.0 --port 8000
```

#### d. Bước deploy:
1. Kết nối repo với Render.
2. Chọn "Web Service" > "Python".
3. Đảm bảo Render nhận đúng:
   - **Build command:** `pip install -r requirements-backend.txt`
   - **Start command:** render sẽ đọc từ Procfile hoặc điền như trên.
4. **Kiểm tra phiên bản Python**: Đảm bảo Render RECOGNIZES `runtime.txt`. Nếu log vẫn dùng Python 3.13:
   - Vào Dashboard > Environment > Add variable: `PYTHON_VERSION` với value `3.11.8`.
5. Chờ hệ thống build; service phải khởi động OK & không báo lỗi scipy/numpy.

### 3. Frontend (Streamlit)
#### a. File cần có:
- `app.py` (hoặc main streamlit file)
- `requirements-frontend.txt` (chỉ các gói Streamlit UI + visualization)
- KHÔNG import/scikit-learn/tensorflow... trừ khi frontend xài trực tiếp
- KHÔNG để dependecy FastAPI/uvicorn ở đây
- KHÔNG trộn requirements-frontend.txt vào backend

#### b. requirements-frontend.txt ví dụ:
```
pip>=25.3
setuptools>=80.9.0
wheel>=0.40.0
streamlit==1.26.0
streamlit-option-menu==0.6.2
pandas==2.1.1
altair==5.2.0
matplotlib==3.7.2
```

#### c. Deploy trên Streamlit Cloud:
- Đẩy code lên Github.
- Deploy repo trên https://streamlit.io/cloud hoặc tương ứng.
- Streaming Cloud sẽ tự build dựa vào `requirements-frontend.txt`.

### 4. Lưu ý chung khi deploy trên Render

- **runtime.txt**: Bắt buộc, ghi rõ python-3.11.8, KHÔNG dùng version mới hơn nếu không chắc chắn.
- **Nếu Render bỏ qua runtime.txt**: vào Dashboard, Add ENV: `PYTHON_VERSION = 3.11.8`.
- **Không trộn dependecy backend (ML, FastAPI) vào frontend (Streamlit)**; mỗi service 1 file requirements riêng biệt.
- Nếu deploy báo lỗi scipy/numpy "metadata-generation-failed" hoặc Python version không khớp, kiểm tra lại 2 bước trên.
- Nếu model/data nặng, chú ý limit 100MB trên Render, cân nhắc lưu external/object storage.

---

## Checklist deploy backend lên Render
- [x] Kiểm tra đủ các file: backend_api.py, requirements-backend.txt, runtime.txt, Procfile, dataset/, results/
- [x] requirements-backend.txt đúng chuẩn (không chứa package thừa)
- [x] runtime.txt với nội dung python-3.11.8
- [x] Procfile hợp lệ cho uvicorn/fastapi
- [x] Environment variable PYTHON_VERSION hoặc kiểm tra log Python version của Render
- [x] Build & deploy, check status/health endpoint

## Checklist deploy frontend (Streamlit)
- [x] Đảm bảo code app.py kết nối đúng backend URL
- [x] requirements-frontend.txt đúng chuẩn (chỉ nhập các package UI/visual)
- [x] Deploy trên Streamlit Cloud
- [x] Test giao diện web, upload ảnh/test API

---
**Liên hệ:** Nếu có lỗi, trước tiên check log deploy và xác thực version Python của Render/service cloud. Đảm bảo không trộn lẫn requirements-backend.txt với requirements-frontend.txt.

**Tham khảo:**  
- https://render.com/docs/python-version  
- https://docs.streamlit.io/
- https://github.com/scipy/scipy/issues/19627
