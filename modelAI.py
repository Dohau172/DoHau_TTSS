import cv2
import numpy as np
import tensorflow.lite as tflite

# 1. Nạp Model từ file bạn đã tải lên
interpreter = tflite.Interpreter(model_path="selfie_segmentation.tflite")
interpreter.allocate_tensors()

# Lấy thông tin đầu vào và đầu ra của model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Lấy kích thước yêu cầu của model (thường là 256x256 cho selfie segmentation)
input_shape = input_details[0]['shape']
height, width = input_shape[1], input_shape[2]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 2. Tiền xử lý hình ảnh
    # Chuyển BGR -> RGB và Resize theo yêu cầu của model
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (width, height))
    
    # Chuẩn hóa (Normalize) về khoảng [0, 1] và thêm chiều batch (1, H, W, 3)
    input_data = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0

    # 3. Chạy Inference (Dự đoán)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # 4. Lấy kết quả Mask
    # Output thường có dạng (1, H, W, 1)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    mask = np.squeeze(output_data) # Bỏ các chiều thừa -> (H, W)

    # 5. Xử lý Mask để hiển thị (Resize lại về kích thước frame gốc)
    mask_visual = (mask > 0.5).astype(np.uint8) * 255
    mask_visual = cv2.resize(mask_visual, (frame.shape[1], frame.shape[0]))

    cv2.imshow('Manual TFLite Mask', mask_visual)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()