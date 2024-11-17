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
import os
from aiortc import RTCConfiguration, RTCIceServer

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Cấu hình RTC với STUN server
RTC_CONFIGURATION = RTCConfiguration([
    RTCIceServer(urls=["stun:stun.l.google.com:19302"])
])

# Tải mô hình đã lưu từ Google Drive
@st.cache_resource
def load_model():
    file_id = '1dK8AzMvw2VyfGOpBxjgMDO1ZFev9Wp-9'  # Thay thế bằng ID tệp Google Drive của bạn
    output = 'SVMmodel_Final.pkl'
    gdown.download(id=file_id, output=output, quiet=False, fuzzy=True)
    
    # Kiểm tra kích thước tệp sau khi tải xuống
    if os.path.exists(output):
        file_size = os.path.getsize(output)
        st.write(f"Kích thước tệp mô hình: {file_size / (1024 * 1024):.2f} MB")
    else:
        st.error("Không thể tải tệp mô hình từ Google Drive.")
        st.stop()
    
    try:
        model = joblib.load(output)
        st.success("Mô hình được tải thành công.")
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}")
        st.stop()
    
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
        # Đảm bảo đường dẫn tương đối tới tệp cascade
        cascade_path = 'models/haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            st.error(f"Không tìm thấy tệp cascade: {cascade_path}")
            st.stop()
        self.faceCascade = cv2.CascadeClassifier(cascade_path)
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
            try:
                prediction = self.model.predict([final_image])[0]
            except Exception as e:
                prediction = "Error"
                st.error(f"Lỗi khi dự đoán cảm xúc: {e}")

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
        async_processing=True,
        rtc_configuration=RTC_CONFIGURATION  # Thêm cấu hình RTC vào đây
    )

if __name__ == "__main__":
    main()
