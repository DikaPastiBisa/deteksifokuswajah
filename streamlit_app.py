import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import time
import base64
from sort import SimpleTracker
import random
from PIL import Image
import matplotlib.pyplot as plt   # <<< INI YANG HILANG TADI

# ==============================================================
# LOAD MODEL
# ==============================================================
model = tf.keras.models.load_model("mobilenetv3_focus_final.keras")
IMG_SIZE = (224, 224)

id2class = {0: "LEFT", 1: "CENTER", 2: "RIGHT"}
counts = {"LEFT": 0, "CENTER": 0, "RIGHT": 0}

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
tracker = SimpleTracker()

# ==============================================================
# RANDOM COLOR PER ID
# ==============================================================
def color_for_id(ID):
    random.seed(ID)
    return (
        random.randint(50, 255),
        random.randint(50, 255),
        random.randint(50, 255)
    )

# ==============================================================
# PREDIKSI WAJAH (ANTI ERROR)
# ==============================================================
def predict_face(face):
    if face is None or face.size == 0:
        return None, None

    if face.shape[0] < 15 or face.shape[1] < 15:
        return None, None

    try:
        img = cv2.resize(face, IMG_SIZE)
    except:
        return None, None

    img = preprocess_input(img.astype("float32"))
    img = np.expand_dims(img, axis=0)

    try:
        pred = model.predict(img, verbose=0)[0]
    except:
        return None, None

    idx = np.argmax(pred)
    return id2class[idx], float(pred[idx])


# ==============================================================
# STREAMLIT UI
# ==============================================================
st.set_page_config(page_title="Deteksi Arah Pandangan", layout="wide")

st.title("🎯 Deteksi Arah Pandangan – Multi-Person (MobileNetV3)")
st.markdown("Real-time multi-person gaze tracking using MobileNetV3 + Streamlit")

col1, col2 = st.columns([2, 1])

frame_window = col1.empty()
chart_window = col2.empty()
status_window = col2.empty()
fps_window = col2.empty()

# ==============================================================
# VIDEO SOURCE (WEBCAM / IP CAMERA / VIDEO FILE)
# ==============================================================
source = st.sidebar.selectbox(
    "Pilih Sumber Kamera:",
    ["Webcam (0)", "Webcam (1)", "IP Camera URL", "Upload Video"]
)

if source == "Webcam (0)":
    cap = cv2.VideoCapture(0)
elif source == "Webcam (1)":
    cap = cv2.VideoCapture(1)
elif source == "IP Camera URL":
    url = st.sidebar.text_input("Masukkan URL IP Camera (rtsp/http):")
    if url:
        cap = cv2.VideoCapture(url)
    else:
        st.warning("Masukkan URL kamera dulu.")
        st.stop()
else:
    uploaded = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi"])
    if uploaded:
        bytes_data = uploaded.read()
        temp_path = "uploaded_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(bytes_data)
        cap = cv2.VideoCapture(temp_path)
    else:
        st.stop()


# ==============================================================
# MAIN LOOP STREAMLIT
# ==============================================================
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Tidak bisa membuka kamera/video.")
        break

    frame = cv2.flip(frame, 1)
    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.2, 5)
    detections = []

    for (x, y, w, h) in faces:
        detections.append([x, y, x+w, y+h])

    tracked = tracker.update(detections)

    fokus_frame = 0
    tidak_frame = 0

    for obj_id, box in tracked:
        x1, y1, x2, y2 = map(int, box)
        face_crop = frame[y1:y2, x1:x2]

        label, conf = predict_face(face_crop)
        if label is None:
            continue

        counts[label] += 1

        if label == "CENTER":
            fokus_frame += 1
        else:
            tidak_frame += 1

        color = color_for_id(obj_id)

        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display,
                    f"ID {obj_id} | {label} ({conf:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

    # STATUS
    if fokus_frame == 0 and tidak_frame == 0:
        status = "Tidak Ada Wajah"
        status_color = "black"
    elif fokus_frame >= tidak_frame:
        status = "FOKUS"
        status_color = "green"
    else:
        status = "TIDAK FOKUS"
        status_color = "red"

    # Pie chart
    fokus = counts["CENTER"]
    tidak = counts["LEFT"] + counts["RIGHT"]
    fig = plt.figure(figsize=(4, 4))
    plt.pie([fokus, tidak], labels=["Fokus", "Tidak Fokus"],
            autopct="%1.1f%%", colors=["#4CAF50", "#F44336"])
    chart_window.pyplot(fig)

    # FPS
    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now

    status_window.markdown(f"### Status: **<span style='color:{status_color}'>{status}</span>**", unsafe_allow_html=True)
    fps_window.markdown(f"### FPS: **{fps:.1f}**")

    # Show video
    rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    frame_window.image(rgb)

cap.release()
