import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
from PIL import Image

# MediaPipeのPoseモデルを初期化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

# Streamlitの設定
st.set_page_config(page_title="姿勢分析アプリ", layout="centered")
st.title("📸 立位姿勢（矢状面）分析アプリ")
st.write("撮影済みの画像をアップロードして姿勢解析を行います。")

# 側面選択
side_option = st.radio("側面を選択", ("左側面", "右側面"))
st.text(f"現在の側面: {side_option}")

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

# 矢状面用のマーカーセット
LEFT_SAGITTAL_MARKERS = [
    "LEFT_EAR", "LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"
]
RIGHT_SAGITTAL_MARKERS = [
    "RIGHT_EAR", "RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"
]

# フレーム処理関数（画像 + 矢状面マーカー描画）
def process_image(image, side):
    image_np = np.array(image)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # PIL画像をOpenCV形式に変換
    results = pose.process(image_rgb)
    height, width, _ = image_np.shape

    if results.pose_landmarks:
        selected_markers = LEFT_SAGITTAL_MARKERS if side == "左側面" else RIGHT_SAGITTAL_MARKERS
        points = {}
        for marker in selected_markers:
            landmark = getattr(mp_pose.PoseLandmark, marker, None)
            if landmark is None:
                continue

            lm = results.pose_landmarks.landmark[landmark]
            cx, cy = int(lm.x * width), int(lm.y * height)
            points[marker] = (cx, cy)
            cv2.circle(image_np, (cx, cy), 12, (255, 255, 255), -1)  # 白縁
            cv2.circle(image_np, (cx, cy), 8, (255, 0, 0), -1)  # 赤点
        
        for i in range(len(selected_markers) - 1):
            if selected_markers[i] in points and selected_markers[i+1] in points:
                cv2.line(image_np, points[selected_markers[i]], points[selected_markers[i+1]], (173, 216, 230), 3)  # ライトブルー

    return Image.fromarray(image_np)

# ビフォーアフター機能
before_image_path = os.path.join(save_path, "before.png")
after_image_path = os.path.join(save_path, "after.png")
comparison_image_path = os.path.join(save_path, "comparison.png")

# 画像アップロード
st.header("📤 ビフォー画像をアップロード")
before_uploaded = st.file_uploader("ビフォー画像を選択", type=["png", "jpg", "jpeg"], key="before")

st.header("📤 アフター画像をアップロード")
after_uploaded = st.file_uploader("アフター画像を選択", type=["png", "jpg", "jpeg"], key="after")

# 画像解析処理
if before_uploaded:
    before_image = Image.open(before_uploaded).convert("RGB")  # RGB変換
    processed_before = process_image(before_image, side_option)
    processed_before.save(before_image_path)
    st.image(processed_before, caption="解析済みビフォー画像", use_column_width=True)
    st.download_button("📥 ビフォー画像をダウンロード", data=open(before_image_path, "rb").read(), file_name="before.png", mime="image/png")

if after_uploaded:
    after_image = Image.open(after_uploaded).convert("RGB")  # RGB変換
    processed_after = process_image(after_image, side_option)
    processed_after.save(after_image_path)
    st.image(processed_after, caption="解析済みアフター画像", use_column_width=True)
    st.download_button("📥 アフター画像をダウンロード", data=open(after_image_path, "rb").read(), file_name="after.png", mime="image/png")

    # ビフォーアフター比較
    if before_uploaded:
        before_np = np.array(processed_before)
        after_np = np.array(processed_after)

        # サイズを統一
        height = min(before_np.shape[0], after_np.shape[0])
        width = min(before_np.shape[1], after_np.shape[1])
        before_np = cv2.resize(before_np, (width, height))
        after_np = cv2.resize(after_np, (width, height))
        
        comparison_image = np.hstack((before_np, after_np))
        comparison_pil = Image.fromarray(cv2.cvtColor(comparison_image, cv2.COLOR_BGR2RGB))
        comparison_pil.save(comparison_image_path)

        st.image(comparison_pil, caption="ビフォーアフター比較", use_column_width=True)
        st.download_button("📥 ビフォーアフター画像をダウンロード", data=open(comparison_image_path, "rb").read(), file_name="comparison.png", mime="image/png")

