import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# === CẤU HÌNH ===
IMG_SIZE = 128
MODEL_PATH = '/home/son/Documents/Globex_Projects/AmericaProject/cnn_model.h5'
CLASS_NAMES = ['Không đạt yêu cầu', 'Đạt yêu cầu']  

# === Load mô hình đã huấn luyện ===
model = load_model(MODEL_PATH)

def predict(img_path):
    try:
        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)[0][0]
        label = CLASS_NAMES[int(pred > 0.5)]
        print(f" Ảnh: {img_path}")
        print(f" Dự đoán: {label} (Xác suất: {pred:.2f})")

    except Exception as e:
        print(f" Lỗi khi xử lý ảnh: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("  Sử dụng đúng cú pháp: python predict_image.py <đường_dẫn_ảnh>")
    else:
        predict(sys.argv[1])
