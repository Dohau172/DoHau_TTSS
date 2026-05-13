import cv2

# 1. Khởi tạo đối tượng VideoCapture. 
# Tham số '0' thường là ID của webcam mặc định trên máy tính.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở Webcam!")
    exit()

print("Đang mở Webcam... Nhấn 'q' để thoát.")

while True:
    # 2. Đọc từng frame từ webcam [cite: 4]
    ret, frame = cap.read()

    # Nếu không đọc được frame (ví dụ: webcam bị ngắt kết nối)
    if not ret:
        print("Không thể nhận frame. Đang thoát...")
        break

    # 3. Hiển thị frame lên cửa sổ 
    #Lật khung hình theo chiều ngang để tạo hiệu ứng gương
    frame = cv2.flip(frame, 1)
    cv2.imshow('Webcam Test - Project Background Blur', frame)

    # 4. Nhấn phím 'q' để thoát khỏi vòng lặp
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5. Giải phóng tài nguyên sau khi xong [cite: 16]
cap.release()
cv2.destroyAllWindows()