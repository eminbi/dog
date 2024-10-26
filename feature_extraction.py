from skimage.feature import hog
import cv2

def extract_hog_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    features = hog(gray, block_norm="L2-Hys")
    return features
