├── config/                         # 환경 설정 파일 저장
├── data/
│   ├── input_videos/               # 원본 비디오 파일 저장
│   ├── processed/                  # 전처리된 데이터 저장
│   └── features/                   # 특징 추출 데이터 저장
├── models/                         # 학습된 머신러닝 모델 저장
├── src/
│   ├── collectors/                 # 데이터 수집 모듈
│   ├── analyzers/                  # 행동 분석 모듈
│   ├── predictors/                 # 성격 및 감정 예측 모듈
│   └── savers/                     # 데이터 저장 모듈
├── tests/                          # 각 모듈 테스트 코드
├── ui/                             # 사용자 인터페이스 관련 코드
├── result_visualizer/              # 결과 시각화 모듈
├── environment.yml                 # Conda 환경 설정 파일
├── requirements.txt                # pip 패키지 설치 파일
└── README.md                       # 프로젝트 설명 파일
