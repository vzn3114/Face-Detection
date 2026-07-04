# Tài Liệu Chú Thích & Giải Thế Kiến Trúc Hệ Thống Giám Sát Khuôn Mặt AI

Tài liệu này giải thích chi tiết từng phần trong Kế hoạch cải cách và Thiết kế hệ thống của dự án Giám sát Khuôn mặt Đám đông (AI Crowd Face Surveillance Console), cung cấp cơ sở lý thuyết và lý do lựa chọn công nghệ cho từng thành phần.

---

## 1. Yêu Cầu Của Người Dùng (User's Requirement) — Giải Thích Chi Tiết

* **Phát hiện khuôn mặt đám đông (Crowd Face Detection):**
  * *Bối cảnh:* Camera giám sát an ninh thực tế luôn phải đối mặt với các cảnh đông người (nhà ga, sân bay, sự kiện công cộng). Các mô hình phát hiện khuôn mặt thông thường thường bỏ sót các mặt ở xa, nhỏ hoặc bị nghiêng.
  * *Mục tiêu:* Định vị toàn bộ hộp bao (Bounding Box) của tất cả khuôn mặt xuất hiện trong khung hình mà không bỏ sót.
* **Phân vùng khuôn mặt (Face Segmentation):**
  * *Bối cảnh:* Chỉ phát hiện bounding box hình chữ nhật là chưa đủ cho các tác vụ nâng cao (như thay đổi đặc điểm mặt, nhận diện sinh trắc học, hoặc phân tích che khuất).
  * *Mục tiêu:* Trích xuất chính xác đường biên (Contour) vùng da mặt, loại bỏ phần tóc, cổ và nền (background).
* **Trải nghiệm thực tế (Local & Real-time Console):**
  * *Bối cảnh:* Hệ thống giám sát cần chạy trực tiếp trên các máy trạm (edge devices) mà không phụ thuộc quá nhiều vào tài nguyên đám mây đắt đỏ.
  * *Mục tiêu:* Đạt tốc độ suy luận (Inference speed) nhanh nhất có thể trên CPU thường, giao diện mượt mà và trực quan hóa dữ liệu thống kê theo thời gian thực.

---

## 2. Các Tính Năng Của Hệ Thống (Features) — Giải Thích Chi Tiết

* **Live Camera Viewport (Khung nhìn an ninh):**
  * *Giải thích:* Hiển thị trực tiếp luồng camera (webcam), video hoặc ảnh tải lên. Giao diện được vẽ đè các vector đồ họa (Bbox, Landmarks, Mask) thông qua thẻ HTML5 canvas đặt song song với thẻ video. Việc vẽ trên canvas giúp giao diện không bị giật lag so với việc backend phải liên tục render ảnh vẽ sẵn rồi gửi về.
* **Face Target Gallery (Thư viện chân dung):**
  * *Giải thích:* Tách biệt các khuôn mặt và hiển thị dưới dạng ảnh PNG trong suốt. Điều này cho phép nhân viên an ninh tập trung vào các "mục tiêu khả nghi" mà không bị phân tâm bởi bối cảnh xung quanh.
* **Real-time Analytics Dashboard (Bảng phân tích trực quan):**
  * *Giải thích:* Thay vì in logs chữ thuần dòng lệnh, hệ thống sử dụng ApexCharts.js vẽ các biểu đồ động:
    * *Biểu đồ Donut:* Thống kê nhanh tỷ lệ phân bổ chất lượng (Excellent, Good, Poor, Unusable) để đánh giá điều kiện ánh sáng và góc quay camera.
    * *Biểu đồ cột ngang (Bar Chart):* Phân tích các góc quay đầu (Head Pose). Nếu camera ghi nhận phần lớn đối tượng có góc "Head Down" hoặc "Head Left/Right" liên tục, có thể camera đang đặt quá cao hoặc sai góc.
    * *Biểu đồ đường (Line Chart):* Thống kê biến động số đối tượng theo thời gian thực để phát hiện các tình huống đám đông tụ tập bất thường (Crowd Alert).
* **Surveillance Alerts & Logs (Nhật ký cảnh báo thông minh):**
  * *Giải thích:* Sử dụng thuật toán Web Audio API để phát âm thanh tần số cao (beep) khi có cảnh báo. Hệ thống tự động ghi nhận các sự kiện như: Mặt quá nhỏ (Too Small), Mặt bị che khuất (Occluded), hoặc Đám đông vượt ngưỡng (Crowd Alarm).

---

## 3. Giải Pháp Công Nghệ (Tech Solutions) — Giải Thích Chi Tiết

* **Tại sao sử dụng FastAPI thay thế Flask?**
  * *Bất đồng bộ (Asynchronous):* FastAPI được xây dựng trên nền tảng Starlette và Uvicorn, hỗ trợ xử lý song song hàng nghìn kết nối đồng thời qua async/await. Đối với ứng dụng stream camera gửi liên tiếp 10-15 ảnh mỗi giây, Flask (đồng bộ) sẽ bị thắt nút cổ chai (blocking), trong khi FastAPI xử lý trơn tru.
  * *Swagger UI tự động:* Truy cập /docs sẽ tự động hiển thị giao diện kiểm thử API chuyên nghiệp giúp debug backend độc lập với frontend dễ dàng.
* **Tại sao sử dụng ONNX Runtime thay thế PyTorch để suy luận (Inference)?**
  * *Tối ưu hóa phần cứng:* ONNX (Open Neural Network Exchange) định nghĩa một đồ thị tính toán thống nhất. ONNX Runtime tối ưu đồ thị này (gộp các lớp toán học, cắt tỉa node dư thừa) để chạy trực tiếp trên tập lệnh CPU (như AVX2/AVX512 trên Intel/AMD) hoặc GPU.
  * *Hiệu suất vượt trội:* Tốc độ suy luận (Inference Latency) nhanh gấp 2 đến 3 lần so với PyTorch CPU gốc.
  * *Nhẹ nhàng & Dễ triển khai:* Máy chủ production chỉ cần cài gói onnxruntime (nặng vài chục MB) thay vì bộ cài đặt PyTorch khổng lồ (nặng 1.5 - 2.0 GB).
* **Mô hình Nhận diện: RetinaFace**
  * RetinaFace là mô hình nhận diện khuôn mặt dạng một giai đoạn (one-stage face detector) cực kỳ mạnh mẽ, sử dụng tính năng kim tự tháp (Feature Pyramid Network - FPN) và liên kết đa nhiệm (multi-task loss) để đồng thời dự đoán: Bounding Box, Điểm tin cậy (Confidence), và 5 điểm mốc Landmark (mắt, mũi, miệng).
  * Mô hình này vượt trội hơn Haar Cascades cũ (lỗi thời, dễ bắt nhầm) và Dlib HOG (chậm trên CPU).
* **Mô hình Phân vùng: U-Net**
  * Cấu trúc hình chữ U đối xứng gồm: Nhánh co rút (Encoder - trích xuất đặc trưng ảnh) và Nhánh mở rộng (Decoder - khôi phục độ phân giải ảnh).
  * Đặc điểm cốt lõi là các đường kết nối tắt (Skip Connections) truyền trực tiếp thông tin biên từ Encoder sang Decoder, giúp mặt nạ phân vùng giữ được độ sắc nét ở đường viền khuôn mặt (không bị mờ nhòe).

---

## 4. Luồng Xử Lý Logic & Trí Tuệ Nhân Tạo (Logic + AI) — Giải Thích Chi Tiết

### Bước 1: Tiền xử lý ảnh (Preprocessing)
Ảnh BGR từ OpenCV được chuẩn hóa bằng cách trừ đi giá trị trung bình kênh màu ImageNet (104, 117, 123) và chuyển đổi sang tensor dạng (1, 3, H, W) trước khi nạp vào RetinaFace ONNX.

### Bước 2: Lọc hộp nhận diện trùng lặp (Non-Maximum Suppression - NMS)
Mô hình RetinaFace sẽ tạo ra hàng ngàn hộp bao giả định (anchors). Giải thuật NMS dựa trên chỉ số giao nhau trên diện tích chung (Intersection over Union - IoU) để giữ lại hộp bao có độ tin cậy cao nhất và loại bỏ các hộp bao trùng lặp bao quanh cùng một khuôn mặt.

### Bước 3: Phân vùng khuôn mặt (Face Segmentation)
* Vùng khuôn mặt sau khi được phát hiện sẽ được cắt ra (Crop).
* Ảnh crop được resize về kích thước chuẩn 512x512 và đưa qua mô hình U-Net ONNX.
* Đầu ra của mô hình U-Net là một tensor có kích thước (19, 512, 512) đại diện cho 19 bộ phận (da, mắt, mày, mũi, môi, tóc, mũ, v.v.).
* Chúng ta gộp 13 bộ phận thuộc về vùng da mặt và ngũ quan (FACE_CLASSES từ 1 đến 13) bằng hàm np.argmax để tạo ra mặt nạ nhị phân (Binary Mask) có giá trị 0 (nền đen) và 255 (vùng mặt trắng).

### Bước 4: Đánh giá chất lượng khuôn mặt (Face Quality Analytics)
Hệ thống sử dụng các phép toán hình học và thống kê trực tiếp trên CPU để đưa ra đánh giá:
1. **Độ che khuất (Occlusion):** 
   * Công thức: Tỷ lệ visibility = (Tổng số pixel vùng mặt có giá trị 255 / Diện tích bounding box) * 100%
   * Nếu tỷ lệ này nhỏ hơn 65%, hệ thống xác định khuôn mặt bị che khuất một phần (do đeo khẩu trang, kính râm lớn hoặc bị tay che) và gán nhãn cảnh báo "Face Occluded".
2. **Góc quay đầu (Head Pose):** 
   Sử dụng khoảng cách giữa 5 điểm Landmarks:
   * **Yaw (Xoay trái/phải):** Tính tỷ lệ khoảng cách từ Mũi tới Mắt trái so với khoảng cách giữa 2 Mắt. Nếu tỉ lệ lệch quá xa mức cân bằng (ngưỡng lệch khỏi 0.5), xác định đối tượng đang quay sang trái hoặc phải.
   * **Pitch (Cúi/Ngửa):** Tính tỷ lệ khoảng cách từ Mắt tới Mũi so với khoảng cách từ Mắt tới Miệng. Tỷ lệ quá thấp xác định đối tượng đang ngửa mặt lên, quá cao xác định đối tượng đang cúi mặt xuống.

### Bước 5: Tạo ảnh trong suốt (Alpha Blending)
Sử dụng hàm Gaussian Blur làm mềm viền mặt nạ nhị phân, sau đó ghép mặt nạ này làm kênh Alpha (kênh trong suốt) vào ảnh RGB gốc để trích xuất ảnh khuôn mặt dạng tròn/oval trong suốt đẹp mắt.

---

## 5. Kế Hoạch Triển Khai (Implement) — Giải Thích Chi Tiết
* **Chuyển đổi ONNX (export_onnx.py):** Viết script xuất mô hình trực tiếp từ PyTorch sang ONNX. Quá trình này cần cấu hình dynamic_axes cho RetinaFace để mô hình có thể nhận đầu vào có kích thước ảnh thay đổi động (H x W) tùy thuộc vào độ phân giải camera đầu vào.
* **Tách biệt Frontend/Backend:** Backend chỉ chịu trách nhiệm nhận ảnh, chạy mô hình AI và trả về kết quả dưới dạng JSON (tọa độ hộp bao, điểm landmarks, chất lượng khuôn mặt) và ảnh cắt dạng chuỗi Base64. Frontend nhận JSON và vẽ đè lên canvas, đồng thời nạp ảnh Base64 vào thư mục Gallery. Cấu trúc này tối ưu băng thông mạng và giúp ứng dụng hoạt động mượt mà.

---

## 6. Kế Hoạch Kiểm Thử (Test) — Giải Thích Chi Tiết
* **Accuracy Test (Độ chính xác):** Đo lường sự sai lệch giá trị đầu ra (Tọa độ, phân lớp) giữa mô hình PyTorch gốc và mô hình ONNX để đảm bảo quá trình lượng tử hóa/xuất file không làm mất độ chính xác của mạng thần kinh.
* **Latency Test (Độ trễ):** Đo lường thời gian từ lúc backend nhận ảnh tới khi trả kết quả. Mục tiêu là duy trì độ trễ nhỏ hơn 120ms trên CPU thông thường để đảm bảo luồng webcam đạt tốc độ tối thiểu từ 10 đến 12 FPS (chuẩn tối thiểu cho camera an ninh giám sát).
