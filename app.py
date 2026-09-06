import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="Real-Time Object Detection", layout="wide")

# Cache model loading so it doesn't reload on every UI click
@st.cache_resource
def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO("yolov8n.pt").to(device)
    return model, device

model, device = load_model()
allowed_classes = [0, 39, 40, 41]  # Person, bottle, wine glass, cup

def process_frame(frame, conf_threshold=0.6):
    results = model(frame, device=device, verbose=False)[0]
    for box in results.boxes:
        conf = float(box.conf[0].item())
        cls_id = int(box.cls[0].item())
        if conf >= conf_threshold and cls_id in allowed_classes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = f"{model.names[cls_id]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

st.title("🔍 Real-Time Object Detection")
st.markdown("### Powered by **YOLOv8 & Streamlit**")

col1, col2 = st.columns(2)
mode = col1.radio("🎥 Choose Input Mode:", ["Image", "Video", "Live Webcam"], index=0)

if mode == "Image":
    uploaded_file = col1.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        processed = process_frame(img_np.copy())
        col2.image(processed, caption="📸 Processed Image", use_column_width=True)

elif mode == "Video":
    uploaded_video = col1.file_uploader("📤 Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = col2.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed = process_frame(frame)
            stframe.image(processed, channels="BGR")
            
        cap.release()
        os.remove(tfile.name)

elif mode == "Live Webcam":
    class YOLOVideoProcessor(VideoTransformerBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            processed = process_frame(img)
            return av.VideoFrame.from_ndarray(processed, format="bgr24")

    col2.markdown("#### Click **START** below to stream your camera:")
    webrtc_streamer(
        key="object-detection",
        video_processor_factory=YOLOVideoProcessor,
        media_stream_constraints={"video": True, "audio": False}
    )
