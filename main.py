# main.py

import os
from data_preprocessing import preprocess_video
from feature_extraction import extract_features
from model_training import train_model
from db_manager import DBManager
import configparser

# Step 1: Config 파일 불러오기
config = configparser.ConfigParser()
config.read('config.json')

# Step 2: 데이터베이스 연결 설정
db = DBManager(config['DATABASE']['db_path'])

# Step 3: 비디오 데이터 전처리
def run_preprocessing():
    video_path = config['PATH']['input_video']
    output_dir = config['PATH']['processed_data']
    preprocess_video(video_path, output_dir)
    print("Step 3: 데이터 전처리가 완료되었습니다.")

# Step 4: 특징 추출 및 데이터베이스 저장
def run_feature_extraction():
    features = extract_features(config['PATH']['processed_data'])
    for feature in features:
        db.save_behavior(feature['behavior'], feature['confidence'])
    print("Step 4: 특징 추출 및 데이터베이스 저장이 완료되었습니다.")

# Step 5: 모델 학습
def run_training():
    data_path = config['PATH']['feature_data']
    train_model(data_path)
    print("Step 5: 모델 학습이 완료되었습니다.")

# Step 6: 예측 및 결과 확인
def run_prediction():
    # 예측 모델 불러오기 및 테스트 데이터 예측 (여기서는 예시로 설명)
    # 예측된 결과를 시각화하거나 콘솔에 출력합니다
    print("Step 6: 모델을 사용한 예측이 완료되었습니다.")
    # 여기에 예측 모델을 활용해 결과를 출력하는 코드 추가 가능

# Step 7: 실행 워크플로
if __name__ == "__main__":
    run_preprocessing()
    run_feature_extraction()
    run_training()
    run_prediction()
    print("모든 분석 과정이 완료되었습니다.")
