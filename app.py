import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

# ==========================================
# 1. CẤU HÌNH (SỬA ĐƯỜNG DẪN CỦA BẠN)
# ==========================================
# File weights (.h5) mà bạn đã test là chạy được với ảnh dataset
WEIGHTS_PATH = r"D:/gr1/FINAL_100_EPOCHS/gen_100.weights.h5"  

IMAGE_SIZE = 128
C_DIM = 3 

# ==========================================
# 2. MODEL (GIỮ NGUYÊN)
# ==========================================
class InstanceNormalization(layers.Layer):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
    def build(self, input_shape):
        self.scale = self.add_weight(name='scale', shape=input_shape[-1:], initializer=tf.random_normal_initializer(1., 0.02), trainable=True)
        self.offset = self.add_weight(name='offset', shape=input_shape[-1:], initializer='zeros', trainable=True)
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        return self.scale * ((x - mean) * tf.math.rsqrt(variance + self.epsilon)) + self.offset

def res_block(x, filters):
    res = x
    x = layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False)(x)
    x = InstanceNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False)(x)
    x = InstanceNormalization()(x)
    return layers.add([res, x])

def build_generator(image_shape=(128, 128, 3), c_dim=3):
    img_input = layers.Input(shape=image_shape)
    label_input = layers.Input(shape=(c_dim,))
    label_layer = layers.RepeatVector(image_shape[0] * image_shape[1])(label_input)
    label_layer = layers.Reshape((image_shape[0], image_shape[1], c_dim))(label_layer)
    x = layers.concatenate([img_input, label_layer])
    x = layers.Conv2D(64, 7, strides=1, padding='same', use_bias=False)(x); x = InstanceNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv2D(128, 4, strides=2, padding='same', use_bias=False)(x); x = InstanceNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv2D(256, 4, strides=2, padding='same', use_bias=False)(x); x = InstanceNormalization()(x); x = layers.ReLU()(x)
    for _ in range(6): x = res_block(x, 256)
    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False)(x); x = InstanceNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False)(x); x = InstanceNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv2D(3, 7, strides=1, padding='same', use_bias=False)(x)
    output = layers.Activation('tanh')(x)
    return Model([img_input, label_input], output)

# ==========================================
# 3. GIAO DIỆN & XỬ LÝ ẢNH (ĐÃ CẬP NHẬT CROP)
# ==========================================
class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Emotion Changer (Auto Face Crop)")
        self.root.geometry("950x600")
        self.root.resizable(False, False)
        
        # Load Haar Cascade để nhận diện khuôn mặt (có sẵn trong cv2)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # --- UI LAYOUT ---
        tk.Label(root, text="DEMO AI: TỰ ĐỘNG CẮT & CHUYỂN CẢM XÚC", font=("Arial", 16, "bold"), bg="#ddd", pady=10).pack(fill=tk.X)
        
        main_frame = tk.Frame(root, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Cột Trái
        frame_in = tk.LabelFrame(main_frame, text="Ảnh Đầu Vào", font=("Arial", 11, "bold"))
        frame_in.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        self.canvas_in = tk.Canvas(frame_in, bg="#eee", width=300, height=300)
        self.canvas_in.pack(pady=20)
        tk.Button(frame_in, text="📂 Chọn Ảnh", command=self.load_image, height=2, bg="#ddd").pack(fill=tk.X, padx=50)

        # Cột Giữa
        frame_mid = tk.Frame(main_frame)
        frame_mid.pack(side=tk.LEFT, padx=10)
        tk.Label(frame_mid, text="Chọn Cảm Xúc:", font=("Arial", 12)).pack(pady=(50, 10))
        self.emo_var = tk.IntVar(value=1)
        modes = [("😐 Bình thường", 0), ("😄 Vui vẻ", 1), ("😢 Buồn bã", 2)]
        for txt, val in modes:
            tk.Radiobutton(frame_mid, text=txt, variable=self.emo_var, value=val, font=("Arial", 12)).pack(anchor="w", pady=5)
        
        self.btn_run = tk.Button(frame_mid, text="👉 CHUYỂN ĐỔI", command=self.process, bg="green", fg="white", font=("Arial", 12, "bold"), height=2, state=tk.DISABLED)
        self.btn_run.pack(pady=30)
        self.lbl_status = tk.Label(frame_mid, text="Đang tải model...", fg="blue")
        self.lbl_status.pack()

        # Cột Phải
        frame_out = tk.LabelFrame(main_frame, text="Kết Quả AI", font=("Arial", 11, "bold"))
        frame_out.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        self.canvas_out = tk.Canvas(frame_out, bg="#eee", width=300, height=300)
        self.canvas_out.pack(pady=20)

        self.cv_img_original = None # Ảnh gốc chưa crop
        self.cv_face_crop = None    # Ảnh mặt đã crop
        
        self.root.after(100, self.init_model)

    def init_model(self):
        try:
            self.generator = build_generator(c_dim=C_DIM)
            if os.path.exists(WEIGHTS_PATH):
                self.generator.load_weights(WEIGHTS_PATH)
                self.lbl_status.config(text="✅ Model sẵn sàng!", fg="green")
            else:
                self.lbl_status.config(text="❌ Không thấy file weight!", fg="red")
        except Exception as e:
            self.lbl_status.config(text="❌ Lỗi Model", fg="red")
            messagebox.showerror("Error", str(e))

    def detect_and_crop_face(self, img):
        """Hàm tự động tìm và cắt khuôn mặt to nhất"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Phát hiện khuôn mặt
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None, None # Không tìm thấy mặt
        
        # Lấy khuôn mặt to nhất (thường là mặt chính)
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        x, y, w, h = faces[0]
        
        # Mở rộng vùng cắt một chút cho đẹp (padding)
        padding = int(w * 0.1) # Lấy thêm 10% viền
        h_img, w_img, _ = img.shape
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_img, x + w + padding)
        y2 = min(h_img, y + h + padding)
        
        cropped_face = img[y1:y2, x1:x2]
        return cropped_face, (x1, y1, x2, y2)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image", "*.jpg *.png *.jpeg")])
        if path:
            # 1. Đọc ảnh gốc
            self.cv_img_original = cv2.imread(path)
            self.cv_img_original = cv2.cvtColor(self.cv_img_original, cv2.COLOR_BGR2RGB)
            
            # 2. Tự động Crop mặt
            face, coords = self.detect_and_crop_face(self.cv_img_original)
            
            if face is not None:
                self.cv_face_crop = face
                self.lbl_status.config(text="✅ Đã tìm thấy khuôn mặt!", fg="green")
                # Hiển thị ảnh đã crop lên Canvas Input
                self.show_image(self.cv_face_crop, self.canvas_in)
            else:
                # Nếu không thấy mặt, dùng luôn ảnh gốc (có thể sẽ bị lỗi nhiễu)
                self.cv_face_crop = self.cv_img_original
                self.lbl_status.config(text="⚠️ Không tìm thấy mặt (Dùng ảnh gốc)", fg="orange")
                self.show_image(self.cv_img_original, self.canvas_in)

            self.canvas_out.delete("all")
            self.btn_run.config(state=tk.NORMAL)

    def show_image(self, img, canvas):
        img_pil = Image.fromarray(img)
        # Resize giữ tỷ lệ để hiển thị đẹp trên canvas 300x300
        img_pil.thumbnail((300, 300))
        tk_img = ImageTk.PhotoImage(img_pil)
        canvas.image = tk_img
        # Căn giữa ảnh
        canvas.create_image(150, 150, image=tk_img)

    def process(self):
        if self.cv_face_crop is None: return
        
        try:
            # 1. Resize ảnh mặt về 128x128 (Chuẩn Input Model)
            img_resized = cv2.resize(self.cv_face_crop, (IMAGE_SIZE, IMAGE_SIZE))
            
            # 2. Chuẩn hóa [-1, 1]
            img_norm = (img_resized / 127.5) - 1.0
            inp = np.expand_dims(img_norm, axis=0)

            # 3. Tạo label
            lbl = np.zeros((1, C_DIM))
            lbl[0, self.emo_var.get()] = 1.0

            # 4. Predict
            gen = self.generator.predict([inp, lbl], verbose=0)

            # 5. Hiển thị
            out = (gen[0] + 1) / 2.0 * 255.0
            out = np.clip(out, 0, 255).astype(np.uint8)
            self.show_image(out, self.canvas_out)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()