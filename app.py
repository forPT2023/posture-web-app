import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Streamlit の設定
st.set_page_config(page_title="姿勢分析アプリ", layout="centered")
st.title("📸 立位姿勢（矢状面）分析アプリ")
st.write("カメラでリアルタイムに姿勢を解析します！")

# 側面選択
side_option = st.radio("側面を選択", ("左側面", "右側面"))
st.text(f"現在の側面: {side_option}")

# MediaPipe の Pose モデルを初期化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

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

# 保存用ディレクトリ
save_path = "captured_images"
os.makedirs(save_path, exist_ok=True)

# 画像の初期化オプション
clear_images = st.checkbox("アプリ起動時に前回の画像を消去する", value=True)
if clear_images:
    for file_name in ["before.png", "after.png", "comparison.png"]:
        file_path = os.path.join(save_path, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)

# ビフォーアフター機能
before_image_path = os.path.join(save_path, "before.png")
after_image_path = os.path.join(save_path, "after.png")
comparison_image_path = os.path.join(save_path, "comparison.png")

# WebRTC を使ったカメラ映像の取得
class PoseEstimationTransformer(VideoTransformerBase):
    def __init__(self):
        self.pose = mp.solutions.pose.Pose()
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img)
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
                if selected_markers[i] in points and selected_markers[i + 1] in points:
                    cv2.line(img, points[selected_markers[i]], points[selected_markers[i + 1]], (173, 216, 230), 3)

        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# WebRTC ストリーム
webrtc_ctx = webrtc_streamer(
    key="posture-analysis",
    video_transformer_factory=PoseEstimationTransformer,
    async_transform=True
)

# 撮影ボタン配置
col1, col2 = st.columns(2)
with col1:
    if st.button("📸 ビフォーを撮影"):
        if webrtc_ctx.video_transformer:
            frame = webrtc_ctx.video_transformer.transform(webrtc_ctx.video_frame)
            Image.fromarray(frame).save(before_image_path)
            st.success("ビフォー画像を撮影しました！")

with col2:
    if st.button("📷 アフターを撮影＆比較"):
        if webrtc_ctx.video_transformer:
            frame = webrtc_ctx.video_transformer.transform(webrtc_ctx.video_frame)
            Image.fromarray(frame).save(after_image_path)
            st.success("アフター画像を撮影しました！")

# 画像表示とダウンロード
before_exists = os.path.exists(before_image_path)
after_exists = os.path.exists(after_image_path)
if before_exists:
    before_image = Image.open(before_image_path)
    st.image(before_image, caption="ビフォー", use_container_width=True)
    st.download_button("📥 ビフォー画像をダウンロード", data=open(before_image_path, "rb").read(), file_name="before.png", mime="image/png")

if after_exists:
    after_image = Image.open(after_image_path)
    st.image(after_image, caption="アフター", use_container_width=True)
    st.download_button("📥 アフター画像をダウンロード", data=open(after_image_path, "rb").read(), file_name="after.png", mime="image/png")

    if before_exists:
        before_np = np.array(before_image)
        after_np = np.array(after_image)

        if before_np.shape != after_np.shape:
            height = min(before_np.shape[0], after_np.shape[0])
            width = min(before_np.shape[1], after_np.shape[1])
            before_np = cv2.resize(before_np, (width, height))
            after_np = cv2.resize(after_np, (width, height))

        comparison_image = np.hstack((before_np, after_np))
        Image.fromarray(comparison_image).save(comparison_image_path)

        comparison_pil = Image.open(comparison_image_path)
        st.image(comparison_pil, caption="ビフォーアフター比較", use_container_width=True)
        st.download_button("📥 ビフォーアフター画像をダウンロード", data=open(comparison_image_path, "rb").read(), file_name="comparison.png", mime="image/png")

