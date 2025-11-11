# Hướng dẫn deploy backend API và web lên Cloud

## I. Chuẩn bị  
- Đã tách file requirements.txt (chỉ dùng cho frontend `app.py` trên Streamlit Cloud)
- Đã tạo file requirements-backend.txt để deploy backend API (FastAPI)
- Đã tạo Procfile cho backend (Railway/Render)

## II. Deploy backend API (FastAPI)
### 1. Chuẩn bị repository  
Đảm bảo repo chứa:
- backend_api.py
- requirements-backend.txt
- Procfile
- dataset/, results/ và các model file cần thiết

### 2. Deploy trên Railway hoặc Render  
- **Railway**:  
  1. Truy cập https://railway.app/new
  2. Kết nối với repo GitHub này
  3. Khi Railway hỏi dependencies, đổi thành:  
     ```
     pip install -r requirements-backend.txt
     ```
  4. Deploy tự động (Railway dùng Procfile sẽ chạy:  
     ```
     web: uvicorn backend_api:app --host 0.0.0.0 --port 8000
     ```
     )
  5. Sau khi deploy thành công, lấy đường dẫn public (ví dụ: `https://fruit-backend-production.up.railway.app`)

- **Render**:  
  Tương tự Railway (chọn Python Build, thiết lập entrypoint là Procfile, dùng requirements-backend.txt để cài packages)

### 3. Lưu ý
- **Deploy thất bại thường do thiếu file model/dataset hoặc sai tên `requirements-backend.txt`.**
- Nếu model nặng quá hoặc có file data >100 MB, cân nhắc chia nhỏ hoặc dùng dịch vụ object storage.

## III. Sửa app.py để kết nối backend cloud
- Mở file app.py và sửa các dòng có `"http://localhost:8000"` thành public backend URL, ví dụ:
  ```python
  BACKEND_API_URL = "https://fruit-backend-production.up.railway.app"
  ```
  Và thay mọi requests.post/get từ localhost sang CLOUD_URL này.

## IV. Deploy frontend Streamlit trên Streamlit Cloud
- Truy cập https://streamlit.io/cloud
- Deploy repo này, giữ nguyên requirements.txt (KHÔNG có fastapi/uvicorn/paython-multipart)
- Streamlit sẽ tự build app.py và tạo giao diện web trên cloud.

## V. Test và hoàn thiện
- Upload ảnh/test nhận diện trên web (Streamlit cloud sẽ gọi backend cloud API).
- Nếu báo lỗi 5xx: kiểm tra log web (log backend cloud hoặc thử call bằng Postman).
- Kiểm tra CORS backend và cross-origin nếu gặp lỗi fetch/network.

---

**Tóm lại:**  
- Backend API chạy trên Railway/Render dùng `requirements-backend.txt`, Procfile, backend_api.py, dataset, results.
- Frontend chạy trên Streamlit Cloud, chỉ cần requirements.txt, app.py.
- Sửa app.py trỏ đúng URL backend cloud.
