
# 🐱🐶 Cat vs Dog Image Classifier using HOG + SVM

A simple computer vision Identifier project that classifies images as **Cat** or **Dog** using:

- 📸 HOG (Histogram of Oriented Gradients) for feature extraction
- 🧠 SVM (Support Vector Machine) for training
- 🧰 OpenCV, scikit-learn, and scikit-image libraries

---

## 📁 Project Structure

```

CatDogClassifier/
├── train\_model.py               # Trains SVM using HOG features
├── predict\_image.py             # Predicts class for a test image
├── svm\_cat\_dog\_model\_hog.pkl    # 🔒 Saved trained model
├── PetImages/                   # Dataset folder
│   ├── Cat/                     # Contains cat images
│   └── Dog/                     # Contains dog images
├── test\_images/                 # Images to test predictions
├── README.md                    # This file

````

---

## 📦 Requirements

Install all required libraries:

```bash
pip install opencv-python scikit-learn scikit-image numpy joblib
````

---

## 🧠 How to Train the Model !!

1. Make sure the dataset folders `PetImages/Cat` and `PetImages/Dog` are filled with images
2. Run the training script:

```bash
python train_model.py
```

This will:

* Load cat and dog images
* Extract HOG features
* Train an SVM classifier
* Save the model as `svm_cat_dog_model_hog.pkl`

---

## 🔍 How to Predict New Image

1. Place an image in the `test_images/` folder
2. Open `predict_image.py` and update the path:

```python
image_path = r"test_images/your_image.jpg"
```

3. Run the script:

```bash
python predict_image.py
```

A window will display the image with the predicted label — **Cat 🐱** or **Dog 🐶**

---

## Example Output

```
✅ Model trained and saved as 'svm_cat_dog_model_hog.pkl'
📊 Classification Report:
              precision    recall  f1-score   support
        Cat       0.88      0.87      0.87       200
        Dog       0.89      0.90      0.89       200
Accuracy: 0.8850
```

---

## 📌 Dataset Source

* [Microsoft Cats vs Dogs Dataset (Kaggle Mirror)](https://www.microsoft.com/en-us/download/details.aspx?id=54765)

---

## What I Learned ✅

* Preprocessing images with OpenCV
* Extracting robust image features using HOG
* Training and evaluating an SVM classifier
* Saving/loading ML models with joblib

---
