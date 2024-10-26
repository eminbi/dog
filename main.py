from data_preprocessing import preprocess_video
from feature_extraction import extract_hog_features
from model_training import train_model
from db_manager import DBManager

if __name__ == "__main__":
    video_path = "data/input_videos/sample_video.mp4"
    preprocess_video(video_path, "data/processed")
    train_model("data/features.csv")
    db = DBManager()
    db.save_behavior("jumping", 0.9)
