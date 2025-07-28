import cv2
import joblib
import numpy as np
from skimage.feature import hog
import os
import sys

# Constants
MODEL_PATH = 'svm_cat_dog_model_hog.pkl'
IMG_SIZE = (128, 128)

# Load the pre-trained model
def load_model(path):
    if not os.path.exists(path):
        print(" Model file not found. Please run the training script first.")
        sys.exit()
    return joblib.load(path)

# Preprocess image and extract HOG features
def preprocess_image(image_path, size=IMG_SIZE):
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError("Image not found or unsupported format.")
    
    img_resized = cv2.resize(img_gray, size)
    features = hog(
        img_resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False
    )
    return features.reshape(1, -1)

# Make prediction and display result
def predict_and_show(image_path, model):
    try:
        features = preprocess_image(image_path)
        prediction = model.predict(features)[0]
        label = "Dog 🐶" if prediction == 1 else "Cat 🐱"

        # Show result on image
        img = cv2.imread(image_path)
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("Prediction", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f" Error: {e}")

# Main function
def main():
    model = load_model(MODEL_PATH)

    # Set your image path here (or ask user for input)
    image_path = r"C:\Users\madha\OneDrive\Desktop\intern\task_3\test_images\download.jpg"
    
    if not os.path.isfile(image_path):
        print(" invalid image path. Please check the file location.")
        return

    predict_and_show(image_path, model)

if __name__ == "__main__":
    main()
