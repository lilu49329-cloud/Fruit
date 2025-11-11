# Hướng dẫn khắc phục lỗi build "metadata-generation-failed" khi deploy trên Render

## Nguyên nhân
- Một số package Python như scipy, numpy... ở các phiên bản mới chưa có wheel cho Python >=3.13 trên môi trường Linux cloud (Render).
- Nếu Render tự động chọn Python 3.13, quá trình install sẽ buộc phải build từ source — nhưng môi trường lại thiếu gfortran, flang, ... dẫn đến lỗi không thể cài package (như log đã chỉ ra).

## Cách giải quyết
1. **Chỉ định rõ phiên bản Python hỗ trợ tốt nhất:**
   - Đảm bảo file `runtime.txt` có dòng:
     ```
     python-3.10.13
     ```
     Hoặc 3.11.x (các bản stable, LTS).
   - Buộc Render dùng đúng version này (nếu Render bỏ qua file runtime.txt, có thể phải thiết lập biến môi trường `PYTHON_VERSION=3.10.13` trong dashboard).

2. **Giữ nguyên hoặc nâng nhẹ phiên bản numpy/scipy (ưu tiên wheel):**
   - Giữ nguyên numpy==1.26.4, scipy==1.12.0 (nếu Python < 3.13), hoặc
   - Nếu bắt buộc dùng Python >= 3.13 thì phải bỏ fix version (`numpy`, `scipy` không ghi `==...`) hoặc hạ version, nhưng *không khuyến khích*.
 
3. **Kiểm tra kỹ file requirements.txt**:
   - Không để version numpy, scipy vượt quá support của phiên bản Python mà bạn deploy.
   - Khuyên dùng các bản phổ biến nhất đã có wheel.

4. **Chú ý:** 
   - Nếu buộc phải dùng Python 3.13 (rất không nên), cần build Fortran compiler, cực kỳ phức tạp trên cloud CI và không nên thực hiện.

## Kết luận & hướng dẫn deploy lại
- **Luôn để file `runtime.txt` với nội dung `python-3.10.13`.**
- Nếu deploy vẫn lỗi Python 3.13 trên Render, vào phần **Environment** của Render Dashboard, **add biến môi trường** `PYTHON_VERSION` thành `3.10.13` hoặc `3.10`.
- Push lại code & deploy lại, log cần hiển thị đúng Python version (3.10.x), nếu còn lỗi vui lòng kiểm tra lại file requirements.txt hoặc liên hệ support của Render.

---
**Tóm tắt lý do:** Render hiện tại dùng Python 3.13 mặc dù file runtime.txt đã chỉ định 3.10.13; nguyên nhân do buildpack của Render dùng version mặc định mới nhất nếu cấu hình phụ không đúng. scipy/numpy các version mới không còn support Python cũ (3.10, 3.11) lâu dài, nên deploy cloud cần dùng version phổ biến, phổ biến nhất là Python 3.10 (năm 2024).

---
**Reference:**  
- https://render.com/docs/python-version  
- https://github.com/scipy/scipy/issues/19627  
└── Trichrutdactrung_out.ipynb  # Notebook xuất kết quả đặc trưng
```

## Mô tả chi tiết từng thành phần

| Thành phần | Công dụng chính |
| --- | --- |
| `app.py` | Xây dựng giao diện Streamlit gồm các tab: trang chủ, nhận diện, tìm kiếm, lịch sử, hướng dẫn, báo cáo. Gửi yêu cầu tới backend và hiển thị kết quả. |
| `backend_api.py` | FastAPI xử lý API `/predict`, `/history`, và phục vụ ảnh tĩnh. Nạp mô hình, thực hiện trích đặc trưng, KNN và lưu lịch sử truy vấn vào SQLite. |
| `train_knn_correlation.py` | Script Python huấn luyện/đánh giá KNN dựa trên đặc trưng có sẵn hoặc ảnh mới. Hữu ích cho việc kiểm thử và cập nhật mô hình. |
| `style.css` | Các quy tắc CSS bổ sung để tuỳ biến giao diện Streamlit (thẻ card, typography, responsive). |
| `requirements.txt` | Danh sách thư viện (numpy, tensorflow, fastapi, streamlit, ...) với phiên bản đã cố định phục vụ chạy và huấn luyện. |
| `DEPLOY_GUIDE.md` | Hướng dẫn triển khai backend và Streamlit trên môi trường cục bộ hoặc server/cloud. |
| `dataset/` | Chứa ảnh trái cây được phân loại theo thư mục con (apple/, banana/, ...). Đây là nguồn dữ liệu cho huấn luyện và tìm kiếm. |
| `results/features.npy`, `results/labels.npy`, `results/paths.npy` | Các file numpy lưu đặc trưng ảnh, nhãn và đường dẫn tương ứng. Được sử dụng khi khởi động hệ thống và huấn luyện KNN. |
| `results/knn_correlation_model.joblib` | Mô hình KNN đã huấn luyện (dùng metric tương quan) cho tác vụ nhận diện. |
| `results/knn_image_search.joblib` | Mô hình KNN khác phục vụ tìm kiếm ảnh tương tự (nếu cần triển khai luồng riêng). |
| `results/confusion_matrix.png`, `results/system_diagram.png` | Hình ảnh báo cáo giúp trình bày hiệu năng và kiến trúc hệ thống. |
| `results/queries/` | Thư mục chứa ảnh truy vấn do người dùng upload thông qua giao diện Streamlit. |
| `query_history.db` | CSDL SQLite lưu timestamp, tên file, nhãn dự đoán và danh sách ảnh tương tự. |
| `project_presentation.ipynb`, `Trichrutdactrung.ipynb`, `Trichrutdactrung_out.ipynb` | Notebook phục vụ trình bày, trích xuất và kiểm thử đặc trưng ngoài hệ thống runtime. |

## Yêu cầu hệ thống và chuẩn bị
- Python ≥ 3.10 và `pip`.
- Khuyến nghị sử dụng môi trường ảo (`venv`, `conda`, ...).
- Máy có GPU sẽ rút ngắn thời gian trích đặc trưng, tuy nhiên hệ thống vẫn vận hành được trên CPU.
- Đảm bảo thư mục `dataset/` và `results/` tồn tại với đầy đủ dữ liệu (đặc trưng, mô hình).

## Thiết lập môi trường

1. **(Tuỳ chọn) Tạo môi trường ảo**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

2. **Cài đặt phụ thuộc**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Kiểm tra TensorFlow**
   ```bash
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```

## Chuẩn bị dữ liệu
- Mỗi loại trái cây cần có một thư mục con trong `dataset/` (ví dụ `dataset/apple/`, `dataset/banana/`, ...).
- Hệ thống đọc cấu trúc thư mục để suy ra nhãn.
- Các file đặc trưng (`features.npy`, `labels.npy`, `paths.npy`) được kỳ vọng sẵn có trong `results/`. Nếu chưa có, cần chạy notebook `Trichrutdactrung.ipynb` hoặc quy trình tương tự để trích xuất.

## Huấn luyện và kiểm thử KNN
Chạy script huấn luyện/tìm kiếm:
```bash
python train_knn_correlation.py
```
Script sẽ:
- Nạp đặc trưng từ `results/`.
- Tạo mô hình KNN với metric tương quan.
- Cung cấp menu tương tác trên terminal để kiểm thử ảnh trong và ngoài dataset.
Các mô hình và log được lưu lại trong `results/`.

## Khởi chạy backend API
```bash
uvicorn backend_api:app --reload --host 0.0.0.0 --port 8000
```
- API chính: `POST /predict` nhận file ảnh và trả về nhãn dự đoán cùng top ảnh tương tự.
- API phụ: `GET /history` trả lịch sử truy vấn, `GET /dataset/{path}` phục vụ ảnh tĩnh từ dataset.

## Khởi chạy giao diện người dùng (Streamlit)
Ở terminal khác:
```bash
streamlit run app.py
```
- Streamlit mặc định mở ở http://localhost:8501.
- Người dùng có thể upload ảnh, xem kết quả nhận diện và tìm kiếm ảnh tương tự.
- Tab “Báo cáo” hiển thị nhanh độ chính xác dựa trên đặc trưng hiện có.

## Luồng hoạt động hệ thống
1. Người dùng truy cập giao diện Streamlit (app.py) và tải ảnh truy vấn.
2. Streamlit gửi yêu cầu `POST /predict` tới backend FastAPI kèm ảnh.
3. Backend trích đặc trưng ảnh bằng MobileNetV2 và dự đoán nhãn thông qua mô hình KNN.
4. Backend tính toán các ảnh tương tự nhất trong dataset (theo metric tương quan).
5. Kết quả dự đoán, danh sách ảnh tương tự và danh sách lớp được trả về cho Streamlit.
6. Backend lưu lịch sử truy vấn (ảnh, nhãn, top tương tự) vào `query_history.db` và ảnh tải lên vào `results/queries/`.
7. Streamlit hiển thị kết quả, cho phép người dùng xem lại lịch sử và báo cáo.

## Quản lý lịch sử và dữ liệu trung gian
- **`query_history.db`**: chứa bảng `history` ghi nhận mọi truy vấn. Có thể mở bằng công cụ SQLite để khai thác thêm.
- **`results/queries/`**: mỗi ảnh truy vấn được lưu với timestamp để phục vụ audit hoặc huấn luyện bổ sung.
- **`results/*.joblib` và `results/*.npy`**: cần được sao lưu khi triển khai để đảm bảo mô hình hoạt động nhất quán.
- **Log và biểu đồ**: `results/confusion_matrix.png`, `results/tim_kiem_L1_log.txt`, ... giúp theo dõi hiệu năng và quá trình tìm kiếm.

## Notebook và tài liệu tham khảo
- `project_presentation.ipynb`: tóm tắt dự án, số liệu và biểu đồ phục vụ báo cáo.
- `Trichrutdactrung.ipynb` & `Trichrutdactrung_out.ipynb`: notebook trích xuất và kiểm thử đặc trưng từ MobileNetV2.
- `DEPLOY_GUIDE.md`: chứa hướng dẫn chi tiết để triển khai hệ thống lên server/cloud (mở port, bảo mật, HTTPS, ...).

## Mở rộng và bảo trì
- Có thể thay thế MobileNetV2 bằng kiến trúc khác để cải thiện độ chính xác.
- Nên định kỳ sao lưu `query_history.db` và thư mục `results/`.
- Xem xét thêm cơ chế xác thực API hoặc phân quyền khi triển khai công khai.
- Tối ưu hoá trải nghiệm người dùng bằng cách chỉnh sửa `style.css` hoặc cập nhật layout trong `app.py`.

## Đóng góp & hỗ trợ
- Tạo GitHub issue hoặc liên hệ nhóm phát triển khi cần hỗ trợ.
- Đề xuất cải tiến, bug và yêu cầu tính năng mới rất được hoan nghênh.

Chúc bạn sử dụng hệ thống hiệu quả!
