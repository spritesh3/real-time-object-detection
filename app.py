import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from PIL import Image
import tempfile
import os
import time

# Load YOLOv8 Model with CUDA if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("yolov8s.pt").to(device)  # Using Small model for faster performance
if device == 'cuda':
    model.half()  # Use half precision for better speed

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to perform object detection with confidence threshold and filtered classes
def detect_objects(image, conf_threshold=0.7):
    image = cv2.resize(image, (640, 480))  # Reduce resolution for faster processing
    results = model(image, device=device)[0]
    allowed_classes = [0, 39, 40, 41]  # Example: Only detecting person, bottle, pen
    filtered_detections = [det for det in results.boxes if det.conf[0] > conf_threshold and det.cls[0].item() in allowed_classes]
    img_with_boxes = results.plot(filtered_detections)
    return img_with_boxes

# Streamlit UI Enhancements
st.set_page_config(page_title="Real-Time Object Detection & Face Recognition", layout="wide")
st.title("🔍 Real-Time Object Detection & Face Recognition")
st.markdown("### Using **YOLOv8, OpenCV & DeepFace** 🧠📷")

col1, col2 = st.columns(2)
mode = col1.radio("🎥 Choose Input Mode:", ["Webcam", "Image", "Video"], index=0)

if mode == "Image":
    uploaded_file = col1.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_np = np.array(image)
        
        img_with_boxes = detect_objects(img_np)
        col2.image(img_with_boxes, caption="📸 Processed Image", use_column_width=True)

elif mode == "Video":
    uploaded_video = col1.file_uploader("📤 Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        stframe = col2.empty()
        frame_skip = 2  # Process every 2nd frame for better speed
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
            
            img_with_boxes = detect_objects(frame)
            stframe.image(img_with_boxes, channels="BGR")
        
        cap.release()
        os.remove(tfile.name)

elif mode == "Webcam":
    st.markdown("#### 🟢 Press 'Start' to open the webcam in a new window")
    if st.button("Start Webcam Detection"):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame_skip = 2  # Process every 2nd frame
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
            
            img_with_boxes = detect_objects(frame)
            cv2.imshow("Real-Time Object Detection", img_with_boxes)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
