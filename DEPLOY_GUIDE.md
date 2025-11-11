# HƯỚNG DẪN TRIỂN KHAI HỆ THỐNG

## 1. Cài đặt môi trường

```bash
pip install -r requirements.txt
```

## 2. Chạy backend API (nếu dùng FastAPI)

```bash
uvicorn backend_api:app --reload --host 0.0.0.0 --port 8000
```

## 3. Chạy giao diện người dùng (Streamlit)

```bash
streamlit run app.py
```

## 4. Truy cập hệ thống
- Mở trình duyệt và truy cập địa chỉ: http://localhost:8501
- Tải ảnh lên để nhận diện và xem kết quả.

## 5. Lưu ý triển khai trên server/cloud
- Cần mở port 8000 (API) và 8501 (Streamlit) trên firewall.
- Có thể dùng dịch vụ như Heroku, Azure, AWS để deploy.
- Đảm bảo các file model và dataset đều có trên server.

## 6. Bảo mật & nâng cấp
- Thêm xác thực API nếu cần.
- Sử dụng HTTPS khi triển khai thực tế.
- Có thể chuyển lưu lịch sử truy vấn sang database (MySQL, MongoDB, ...).

---

**Mọi thắc mắc vui lòng liên hệ nhóm phát triển!**
