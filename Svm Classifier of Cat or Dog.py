import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Configuration
IMG_SIZE = (128, 128)
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "visualize": False
}
MODEL_SAVE_PATH = 'svm_cat_dog_model_hog.pkl'
LIMIT_PER_CLASS = 1000
RANDOM_STATE = 42

# Feature Extraction 
def extract_hog_features(image_path, img_size=IMG_SIZE):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, img_size)
    features = hog(img, **HOG_PARAMS)
    return features

#  Dataset Loader
def load_dataset(cat_folder, dog_folder, limit=LIMIT_PER_CLASS):
    X, y = [], []

    def load_class(folder, label):
        print(f" Loading {'Cat' if label == 0 else 'Dog'} images from {folder}...")
        count = 0
        for file in os.listdir(folder):
            if count >= limit:
                break
            path = os.path.join(folder, file)
            features = extract_hog_features(path)
            if features is not None:
                X.append(features)
                y.append(label)
                count += 1

    load_class(cat_folder, 0)
    load_class(dog_folder, 1)

    print(f" Loaded {len(X)} samples.")
    return np.array(X), np.array(y)

# Main Training Function 
def train_model(cat_dir, dog_dir):
    X, y = load_dataset(cat_dir, dog_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    print("\n Training SVM model...")
    model = SVC(kernel='linear', C=1.0)
    model.fit(X_train, y_train)

    print("\n Evaluation!:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(f" Accuracy:{accuracy_score(y_test, y_pred):.4f}")

    joblib.dump(model, MODEL_SAVE_PATH)
    print(f" Model saved to: {MODEL_SAVE_PATH}")

# Entry Point
if __name__ == "__main__":
    cat_dir = r"C:\Users\madha\OneDrive\Desktop\intern\task_3\PetImages\Cat"
    dog_dir = r"C:\Users\madha\OneDrive\Desktop\intern\task_3\PetImages\Dog"

    if not os.path.exists(cat_dir) or not os.path.exists(dog_dir):
        print(" Dataset folders been not found. Please check the paths.")
    else:
        train_model(cat_dir, dog_dir)
