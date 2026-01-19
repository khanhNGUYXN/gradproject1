import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import re # Thư viện để xử lý sắp xếp số

# ==========================================
# 1. CẤU HÌNH (SỬA ĐƯỜNG DẪN FILE .NPZ)
# ==========================================
# Trỏ đúng vào file .npz đang bị lỗi của bạn
NPZ_PATH = r"C:/Users/eclas/Downloads/universal_weights.npz"

IMAGE_SIZE = 128
C_DIM = 3 

# ==========================================
# 2. KIẾN TRÚC MODEL (GIỮ NGUYÊN)
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
# 3. HÀM LOAD NPZ (FIX LỖI SORTING)
# ==========================================
def natural_sort_key(s):
    """
    Hàm này giúp sắp xếp w_1, w_2, w_10 đúng thứ tự.
    Thay vì w_1, w_10, w_2 như mặc định.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def load_weights_from_npz_smart(model, filepath):
    print(f"🔄 Đang đọc file NPZ: {filepath}")
    try:
        with np.load(filepath) as data:
            # Lấy danh sách key
            all_files = data.files
            
            # SẮP XẾP TỰ NHIÊN (QUAN TRỌNG NHẤT)
            sorted_keys = sorted(all_files, key=natural_sort_key)
            
            weights_list = [data[k] for k in sorted_keys]
            
            print(f"   Tìm thấy {len(weights_list)} weights trong file.")
            print(f"   Model cần {len(model.get_weights())} weights.")
            
            # Set weights
            model.set_weights(weights_list)
            
        print(f"✅ Đã nạp thành công! (Natural Sort)")
        return True
    except ValueError as ve:
        print(f"❌ Lỗi Shape: {ve}")
        print("   -> Có thể file NPZ bị thiếu lớp hoặc kiến trúc bị lệch quá nhiều.")
        return False
    except Exception as e:
        print(f"❌ Lỗi khác: {e}")
        return False

# ==========================================
# 4. GIAO DIỆN APP
# ==========================================
class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Emotion Changer (NPZ Fix)")
        self.root.geometry("950x650")
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # UI Header
        tk.Label(root, text="DEMO AI - FIX NPZ", font=("Arial", 16, "bold"), bg="#444", fg="white", pady=10).pack(fill=tk.X)
        main_frame = tk.Frame(root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Trái: Input
        frame_in = tk.LabelFrame(main_frame, text="Input", font=("Arial", 10, "bold"))
        frame_in.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas_in = tk.Canvas(frame_in, bg="#ccc", width=300, height=300)
        self.canvas_in.pack(pady=20)
        tk.Button(frame_in, text="📂 Chọn Ảnh", command=self.load_image, height=2).pack(fill=tk.X, padx=20)

        # Giữa: Control
        frame_mid = tk.Frame(main_frame)
        frame_mid.pack(side=tk.LEFT, padx=10)
        tk.Label(frame_mid, text="Chọn Cảm Xúc:", font=("Arial", 12)).pack(pady=(60, 10))
        self.emo_var = tk.IntVar(value=1)
        # ID chuẩn: 0:Neutral, 1:Happy, 2:Sad
        modes = [("😐 Bình thường", 0), ("😄 Vui vẻ", 1), ("😢 Buồn bã", 2)]
        for txt, val in modes:
            tk.Radiobutton(frame_mid, text=txt, variable=self.emo_var, value=val, font=("Arial", 11)).pack(pady=5, anchor="w")

        self.btn_run = tk.Button(frame_mid, text="⚡ CHUYỂN ĐỔI", command=self.process, bg="green", fg="white", font=("Arial", 12, "bold"), height=2, state=tk.DISABLED)
        self.btn_run.pack(pady=30)
        self.lbl_status = tk.Label(frame_mid, text="...", fg="blue", wraplength=180)
        self.lbl_status.pack(side=tk.BOTTOM, pady=20)

        # Phải: Output
        frame_out = tk.LabelFrame(main_frame, text="Output", font=("Arial", 10, "bold"))
        frame_out.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas_out = tk.Canvas(frame_out, bg="#ccc", width=300, height=300)
        self.canvas_out.pack(pady=20)

        self.cv_face = None
        self.root.after(100, self.init_model)

    def init_model(self):
        self.lbl_status.config(text="Đang nạp Model từ NPZ...")
        try:
            self.generator = build_generator(c_dim=C_DIM)
            if os.path.exists(NPZ_PATH):
                # Dùng hàm load thông minh
                if load_weights_from_npz_smart(self.generator, NPZ_PATH):
                    self.lbl_status.config(text="✅ Model sẵn sàng!", fg="green")
                else:
                    self.lbl_status.config(text="❌ Lỗi load NPZ (Xem terminal)", fg="red")
            else:
                self.lbl_status.config(text="❌ Không thấy file .npz", fg="red")
        except Exception as e:
            self.lbl_status.config(text=f"Lỗi: {e}", fg="red")

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image", "*.jpg *.png *.jpeg")])
        if path:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Auto Crop nhạy hơn
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.05, 3)
            
            if len(faces) > 0:
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                x, y, w, h = faces[0]
                p = int(w * 0.15)
                y1, y2 = max(0, y-p), min(img.shape[0], y+h+p)
                x1, x2 = max(0, x-p), min(img.shape[1], x+w+p)
                self.cv_face = img[y1:y2, x1:x2]
                self.lbl_status.config(text="✅ Đã tìm thấy mặt", fg="green")
            else:
                self.cv_face = img
                self.lbl_status.config(text="⚠️ Dùng ảnh gốc", fg="orange")
            
            pil_img = Image.fromarray(self.cv_face)
            pil_img.thumbnail((300, 300))
            tk_img = ImageTk.PhotoImage(pil_img)
            self.canvas_in.image = tk_img
            self.canvas_in.create_image(150, 150, image=tk_img)
            self.btn_run.config(state=tk.NORMAL)

    def process(self):
        if self.cv_face is None: return
        try:
            img = cv2.resize(self.cv_face, (IMAGE_SIZE, IMAGE_SIZE))
            img = (img / 127.5) - 1.0
            inp = np.expand_dims(img, axis=0)
            lbl = np.zeros((1, C_DIM))
            lbl[0, self.emo_var.get()] = 1.0
            
            out = self.generator.predict([inp, lbl], verbose=0)
            
            out = (out[0] + 1) / 2.0 * 255.0
            out = np.clip(out, 0, 255).astype(np.uint8)
            
            pil_img = Image.fromarray(out)
            pil_img.thumbnail((300, 300))
            tk_img = ImageTk.PhotoImage(pil_img)
            self.canvas_out.image = tk_img
            self.canvas_out.create_image(150, 150, image=tk_img)
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()