import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

# Load mô hình đã huấn luyện
model = keras.models.load_model("fruit_classification_model.h5")
print(model.summary())

# Lấy danh sách nhãn (tên trái cây)
data_dir = "data/train"
class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

# Cấu hình thư mục upload ảnh
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Hàm xử lý ảnh
def preprocess_image(img_path, img_size=(100, 100)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img / 255.0 
    img = np.expand_dims(img, axis=0)  
    return img

# Route trang chủ
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Kiểm tra nếu có file ảnh được upload
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected")

        # Lưu file vào thư mục tạm thời
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Tiền xử lý ảnh và dự đoán
        img = preprocess_image(file_path)
        predictions = model.predict(img)
        print(predictions)
        predicted_class = np.argmax(predictions)
        print(predicted_class)
        predicted_label = class_names[predicted_class]
        print(predicted_label)
        confidence = np.max(predictions)

        return render_template("index.html", uploaded_image=file_path, label=predicted_label, confidence=confidence)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
