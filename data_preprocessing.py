import cv2
import os
import json
import numpy as np

def load_config():
    """Load the configuration from config.json."""
    with open("config.json", "r") as f:
        return json.load(f)

def preprocess_frame(frame, config):
    """Apply preprocessing steps to a video frame based on the config."""
    # Resize the frame if specified in config
    if config['preprocessing'].get('resize', False):
        frame = cv2.resize(frame, (config["frame_width"], config["frame_height"]))
    # Convert to grayscale if specified
    if config['preprocessing'].get('gray_scale', False):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Normalize if specified
    if config['preprocessing'].get('normalize', False):
        frame = (frame / 255.0).astype(np.float32)  # Normalize to 0-1 range
    # Convert back to uint8 if required for saving
    if frame.dtype != np.uint8:
        frame = (frame * 255).astype(np.uint8)
    return frame

def preprocess_video(video_path, output_dir):
    """Process video and save frames based on config settings."""
    config = load_config()  # Load configuration settings
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_dir, exist_ok=True)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Apply preprocessing based on config
        processed_frame = preprocess_frame(frame, config)
        cv2.imwrite(f"{output_dir}/frame_{frame_count:04d}.jpg", processed_frame)
        frame_count += 1

    cap.release()
    print(f"Processed {frame_count} frames and saved to {output_dir}")

# Example usage
if __name__ == "__main__":
    video_path = "input_videos/sample_video.mp4"
    output_dir = "processed_frames"
    preprocess_video(video_path, output_dir)
