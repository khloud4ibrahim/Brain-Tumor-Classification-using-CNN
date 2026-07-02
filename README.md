#  🧠 Brain Tumor Classification using CNN

A deep learning project for classifying brain MRI images into four categories using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

## 📌 Project Overview

This project aims to classify brain MRI scans into one of four classes:

* Glioma
* Meningioma
* Pituitary Tumor
* No Tumor

The model was trained using MRI images resized to **224×224 pixels** and achieved promising classification performance.

---

## 📂 Dataset Structure

```text
dataset/
│
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Each class contains approximately **1400 MRI images**.

---

## ⚙️ Data Preprocessing

The preprocessing pipeline includes:

* Loading MRI images
* Converting images to RGB format
* Resizing images to **224 × 224**
* Normalizing pixel values to the range **[0,1]**
* Converting labels using One-Hot Encoding
* Splitting the dataset into:

  * Training Set
  * Validation Set
  * Testing Set

---

## 🏗️ CNN Architecture

The model consists of:

* Conv2D (32 filters)
* MaxPooling2D
* Conv2D (64 filters)
* MaxPooling2D
* Conv2D (128 filters)
* MaxPooling2D
* Flatten Layer
* Dense Layer (128 neurons)
* Dropout (0.5)
* Output Layer (4 neurons, Softmax)

---

## 🧠 Training Configuration

* Optimizer: Adam
* Loss Function: Categorical Crossentropy
* Batch Size: 32
* Epochs: 10
* Input Size: 224×224×3

---

## 📊 Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* OpenCV
* Pillow
* Matplotlib
* Scikit-learn

---

## 💾 Model Saving

The trained model is saved using:

```python
model.save("brain_tumor_cnn.keras")
```


