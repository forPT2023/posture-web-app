import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from PIL import Image

# MediaPipe の初期化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Streamlit の設定
st.set_page_config(page_title="姿勢分析アプリ", layout="centered")
st.title("📸 立位姿勢（矢状面）分析アプリ")
st.write("ブラウザでカメラを使用して、リアルタイムに姿勢を解析します！")

# 側面選択
side_option = st.radio("側面を選択", ("左側面", "右側面"))
st.text(f"現在の側面: {side_option}")

# 矢状面用のマーカーセット
LEFT_SAGITTAL_MARKERS = [
    mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE
]
RIGHT_SAGITTAL_MARKERS = [
    mp_pose.PoseLandmark.RIGHT_EAR, mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.RIGHT_ANKLE
]

# WebRTC のクラス
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        height, width, _ = img.shape

        if results.pose_landmarks:
            selected_markers = LEFT_SAGITTAL_MARKERS if side_option == "左側面" else RIGHT_SAGITTAL_MARKERS
            points = {}

            for marker in selected_markers:
                lm = results.pose_landmarks.landmark[marker]
                cx, cy = int(lm.x * width), int(lm.y * height)
                points[marker] = (cx, cy)
                cv2.circle(img, (cx, cy), 12, (255, 255, 255), -1)  # 白縁
                cv2.circle(img, (cx, cy), 8, (255, 0, 0), -1)  # 赤点
            
            for i in range(len(selected_markers) - 1):
                if selected_markers[i] in points and selected_markers[i+1] in points:
                    cv2.line(img, points[selected_markers[i]], points[selected_markers[i+1]], (173, 216, 230), 3)

        return frame.from_ndarray(img, format="bgr24")

# WebRTC を使ったカメラストリーミング
webrtc_streamer(key="姿勢解析", video_processor_factory=VideoProcessor)


