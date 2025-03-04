import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
from PIL import Image

# MediaPipeのPoseモデルを初期化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Streamlitの設定
st.set_page_config(page_title="姿勢分析アプリ", layout="centered")
st.title("📸 立位姿勢（矢状面）分析アプリ")
st.write("ブラウザでカメラを使用して、リアルタイムに姿勢を解析します！")

# 側面選択
side_option = st.radio("側面を選択", ("左側面", "右側面"))
st.text(f"現在の側面: {side_option}")

# 保存用ディレクトリ
save_path = "captured_images"
os.makedirs(save_path, exist_ok=True)

# カメラ入力
image_file = st.camera_input("📷 カメラで写真を撮影")

# ビフォーアフター画像パス
before_image_path = os.path.join(save_path, "before.png")
after_image_path = os.path.join(save_path, "after.png")
comparison_image_path = os.path.join(save_path, "comparison.png")

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

# 撮影ボタン配置
col1, col2 = st.columns(2)

# 画像処理関数
def process_frame(image):
    """ 撮影した画像を解析し、関節マーカーを描画する """
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = pose.process(frame)
    height, width, _ = frame.shape

    if results.pose_landmarks:
        selected_markers = LEFT_SAGITTAL_MARKERS if side_option == "左側面" else RIGHT_SAGITTAL_MARKERS
        points = {}

        for marker in selected_markers:
            lm = results.pose_landmarks.landmark[marker]
            cx, cy = int(lm.x * width), int(lm.y * height)
            points[marker] = (cx, cy)
            cv2.circle(frame, (cx, cy), 12, (255, 255, 255), -1)  # 白縁
            cv2.circle(frame, (cx, cy), 8, (255, 0, 0), -1)  # 赤点
        
        for i in range(len(selected_markers) - 1):
            if selected_markers[i] in points and selected_markers[i+1] in points:
                cv2.line(frame, points[selected_markers[i]], points[selected_markers[i+1]], (173, 216, 230), 3)  # ライトブルー

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)

# 📸 ビフォー画像撮影
with col1:
    if st.button("📸 ビフォーを撮影"):
        if image_file is not None:
            before_image = process_frame(Image.open(image_file))
            before_image.save(before_image_path)
            st.success("✅ ビフォー画像を保存しました！")

# 📷 アフター画像撮影
with col2:
    if st.button("📷 アフターを撮影＆比較"):
        if image_file is not None:
            after_image = process_frame(Image.open(image_file))
            after_image.save(after_image_path)
            st.success("✅ アフター画像を保存しました！")

# 画像表示と比較
before_exists = os.path.exists(before_image_path)
after_exists = os.path.exists(after_image_path)

if before_exists:
    before_image = Image.open(before_image_path)
    st.image(before_image, caption="📸 ビフォー画像", use_column_width=True)
    st.download_button("📥 ビフォー画像をダウンロード", data=open(before_image_path, "rb").read(), file_name="before.png", mime="image/png")

if after_exists:
    after_image = Image.open(after_image_path)
    st.image(after_image, caption="📷 アフター画像", use_column_width=True)
    st.download_button("📥 アフター画像をダウンロード", data=open(after_image_path, "rb").read(), file_name="after.png", mime="image/png")

    # ビフォーアフター比較
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
        st.image(comparison_pil, caption="📊 ビフォーアフター比較", use_column_width=True)
        st.download_button("📥 ビフォーアフター画像をダウンロード", data=open(comparison_image_path, "rb").read(), file_name="comparison.png", mime="image/png")

