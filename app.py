import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
from PIL import Image

# MediaPipeのPoseモデルを使用
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Streamlitの設定
st.set_page_config(page_title="姿勢分析アプリ", layout="centered")
st.title("📸 立位姿勢（矢状面）分析アプリ")
st.write("カメラを使用して姿勢を解析します！")

# 側面選択
side_option = st.radio("側面を選択", ("左側面", "右側面"))
st.text(f"現在の側面: {side_option}")

# 保存用ディレクトリ
save_path = "captured_images"
os.makedirs(save_path, exist_ok=True)

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

# カメラ入力
image_file = st.camera_input("📸 カメラで撮影してください")

# 画像処理関数
def process_image(image):
    image = Image.open(image)
    image = np.array(image)

    # OpenCV用にBGR変換
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # MediaPipeで姿勢解析
    results = pose.process(image)
    height, width, _ = image.shape

    if results.pose_landmarks:
        selected_markers = LEFT_SAGITTAL_MARKERS if side_option == "左側面" else RIGHT_SAGITTAL_MARKERS
        points = {}

        for marker in selected_markers:
            lm = results.pose_landmarks.landmark[marker]
            cx, cy = int(lm.x * width), int(lm.y * height)
            points[marker] = (cx, cy)
            cv2.circle(image, (cx, cy), 12, (255, 255, 255), -1)  # 白縁
            cv2.circle(image, (cx, cy), 8, (255, 0, 0), -1)  # 赤点
        
        for i in range(len(selected_markers) - 1):
            if selected_markers[i] in points and selected_markers[i+1] in points:
                cv2.line(image, points[selected_markers[i]], points[selected_markers[i+1]], (173, 216, 230), 3)

    return image

# 撮影画像がある場合に処理
if image_file:
    processed_image = process_image(image_file)
    st.image(processed_image, caption="解析結果", use_column_width=True)

    # 撮影ボタン
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📸 ビフォーを保存"):
            Image.fromarray(processed_image).save(os.path.join(save_path, "before.png"))
            st.success("ビフォー画像を保存しました！")

    with col2:
        if st.button("📷 アフターを保存＆比較"):
            Image.fromarray(processed_image).save(os.path.join(save_path, "after.png"))
            st.success("アフター画像を保存しました！")

# ビフォーアフター画像表示
before_path = os.path.join(save_path, "before.png")
after_path = os.path.join(save_path, "after.png")
if os.path.exists(before_path):
    before_image = Image.open(before_path)
    st.image(before_image, caption="ビフォー", use_column_width=True)

if os.path.exists(after_path):
    after_image = Image.open(after_path)
    st.image(after_image, caption="アフター", use_column_width=True)

    # ビフォーアフター比較
    before_np = np.array(before_image)
    after_np = np.array(after_image)

    if before_np.shape == after_np.shape:
        comparison_image = np.hstack((before_np, after_np))
        st.image(comparison_image, caption="ビフォーアフター比較", use_column_width=True)
    else:
        st.warning("ビフォーアフター画像のサイズが異なるため、比較できません。")

