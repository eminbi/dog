import os
import json
import cv2
import numpy as np
from skimage.feature import hog
import tensorflow as tf
from tensorflow.keras import layers, models

# JSON 파일 읽기
def load_config(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, 'r') as file:
        return json.load(file)

config = load_config('config.json')

# 전처리 함수
def preprocess_frame(frame, config):
    if config['preprocessing'].get('resize', False):
        frame = cv2.resize(frame, tuple(config['model']['input_shape'][:2]))
    if config['preprocessing'].get('gray_scale', False):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if config['preprocessing'].get('normalize', False):
        frame = frame / 255.0
    if config['preprocessing'].get('denoise', False):
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
    if config['preprocessing'].get('rotation_correction', False):
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame

# HOG 특징 추출 함수
def extract_hog_features(frame, config):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    block_norm = config['hog'].get('block_norm', 'L2-Hys')
    features, hog_image = hog(
        gray_frame, visualize=True, block_norm=block_norm
    )
    return features

# CNN 모델 생성 함수
def build_model(config):
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

# 비디오 처리 및 학습 루프
def process_and_train(video_path, config):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    # 비디오 파일 읽기
    video_capture = cv2.VideoCapture(video_path)
    frames = []
    labels = []  # 필요 시 라벨 추가

    # 프레임 처리
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # 전처리
        processed_frame = preprocess_frame(frame, config)
        frames.append(processed_frame)
        
        # HOG 특징 추출
        if config['hog'].get('use_hog', False):
            features = extract_hog_features(processed_frame, config)

        # 실시간 시각화 (프레임마다 표시하지 않음)
        if config['visualization'].get('enable_real_time_visualization', False) and frame_count % 10 == 0:
            cv2.imshow('Processed Frame', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        frame_count += 1

    video_capture.release()
    cv2.destroyAllWindows()

    # CNN 모델 학습
    frames = np.array(frames)
    model = build_model(config)
    batch_size = config['model'].get('batch_size', 32)
    epochs = config['model'].get('epochs', 10)
    
    if len(labels) == 0:
        labels = frames  # 임시로 자기 예측

    model.fit(frames, labels, epochs=epochs, batch_size=batch_size)

# 실행
video_path = config['video_processing']['video_path']
process_and_train(video_path, config)
