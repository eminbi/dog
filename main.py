import cv2
import os
import json
import numpy as np
from skimage.feature import hog
import tensorflow as tf
from tensorflow.keras import layers, models

def load_config(file_path="config.json"):
    with open(file_path, 'r') as file:
        return json.load(file)

def preprocess_frame(frame, config):
    """Apply preprocessing steps to a video frame based on the config."""
    if config['preprocessing'].get('resize', False):
        frame = cv2.resize(frame, tuple(config["model"]["input_shape"][:2]))
    if config['preprocessing'].get('gray_scale', False):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if config['preprocessing'].get('normalize', False):
        frame = (frame / 255.0).astype(np.float32)  # Normalize to 0-1 range
    # Convert to uint8 if required for OpenCV compatibility
    if frame.dtype != np.uint8:
        frame = (frame * 255).astype(np.uint8)
    return frame

def extract_hog_features(frame, config):
    """Extract HOG features from a video frame."""
    if frame.dtype != np.uint8:
        frame = (frame * 255).astype(np.uint8)  # Ensure frame is in uint8 format
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    block_norm = config['hog'].get('block_norm', 'L2-Hys')
    features, hog_image = hog(
        gray_frame, visualize=True, block_norm=block_norm
    )
    return features

def build_model(config):
    """Build a CNN model based on config settings."""
    input_shape = tuple(config['model'].get('input_shape', (224, 224, 3)))
    dropout_rate = config['model'].get('dropout_rate', 0.5)
    optimizer = config['model'].get('optimizer', 'adam')

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def process_and_train(video_path, config):
    """Process video frames and train the CNN model."""
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    video_capture = cv2.VideoCapture(video_path)
    frames = []

    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        processed_frame = preprocess_frame(frame, config)
        frames.append(processed_frame)
        
        if config['hog'].get('use_hog', False):
            features = extract_hog_features(processed_frame, config)

        frame_count += 1

    video_capture.release()
    frames = np.array(frames)
    model = build_model(config)
    batch_size = config['model'].get('batch_size', 32)
    epochs = config['model'].get('epochs', 10)
    model.fit(frames, frames, epochs=epochs, batch_size=batch_size)

if __name__ == "__main__":
    config = load_config()
    video_path = config['video_processing']['video_path']
    process_and_train(video_path, config)
