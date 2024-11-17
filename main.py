import asyncio
import sys
import streamlit as st
import cv2
import numpy as np
import av
from skimage.feature import hog
import joblib
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
import gdown

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Tải mô hình đã lưu từ Google Drive
@st.cache_resource
def load_model():
    url = 'https://drive.google.com/uc?export=download&id=1dK8AzMvw2VyfGOpBxjgMDO1ZFev9Wp-9'
    output = 'SVMmodel_Final.pkl'
    gdown.download(url, output, quiet=False)
    model = joblib.load(output)
    return model

# Sử dụng mô hình
model = load_model()

# Hàm thay đổi kích thước ảnh
def resize_image(image, size):
    return cv2.resize(image, (size, size))

# Hàm trích xuất đặc trưng
def extract_feature_final(image_test):
    im = resize_image(image_test, 64)
    fd1 = hog(im, orientations=7, pixels_per_cell=(8, 8),
              cells_per_block=(4, 4), block_norm='L2-Hys', transform_sqrt=False)
    return fd1

class EmotionRecognizer(VideoProcessorBase):
    def __init__(self):
        self.faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model = model

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Lật hình ảnh theo trục ngang
        img = cv2.flip(img, 1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        faces = self.faceCascade.detectMultiScale(equalized, scaleFactor=1.1, minNeighbors=4)

        for (x, y, w, h) in faces:
            roi_gray = equalized[y:y+h, x:x+w]

            # Trích xuất đặc trưng và dự đoán cảm xúc
            final_image = extract_feature_final(roi_gray)
            prediction = self.model.predict([final_image])[0]

            # Vẽ hình chữ nhật và hiển thị cảm xúc
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, prediction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("Nhận Diện Cảm Xúc Khuôn Mặt Real-Time")

    webrtc_streamer(
        key="emotion-recognition",
        video_processor_factory=EmotionRecognizer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

if __name__ == "__main__":
    main()
