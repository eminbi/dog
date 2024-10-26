import cv2
import os
import json

def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

def preprocess_video(video_path, output_dir):
    config = load_config()
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_dir, exist_ok=True)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (config["frame_width"], config["frame_height"]))
        cv2.imwrite(f"{output_dir}/frame_{frame_count:04d}.jpg", frame)
        frame_count += 1
    cap.release()
