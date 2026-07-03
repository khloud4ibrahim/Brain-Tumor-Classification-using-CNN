#   🧠 Brain Tumor Classification using CNN

A Deep Learning project for classifying brain MRI images into four categories using a Convolutional Neural Network (CNN) built with TensorFlow and Keras, with an interactive Streamlit application for predictions.

## 📌 Project Overview

This project classifies brain MRI scans into one of four classes:

* Glioma
* Meningioma
* No Tumor
* Pituitary Tumor

The model was trained on MRI images resized to **224×224 pixels** and deployed using **Streamlit** for real-time predictions.

---


Each class contains approximately **1400 MRI images**.

---

## ⚙️ Data Preprocessing

* Convert images to RGB
* Resize images to **224 × 224**
* Normalize pixel values to **[0,1]**
* One-Hot Encode labels
* Split data into training, validation, and testing sets

---

## 🏗️ CNN Architecture

* Conv2D (32 filters)
* MaxPooling2D
* Conv2D (64 filters)
* MaxPooling2D
* Conv2D (128 filters)
* MaxPooling2D
* Flatten Layer
* Dense Layer (128 neurons)
* Dropout (0.5)
* Dense Layer (4 neurons, Softmax)

---

## 🧠 Training Configuration

* Optimizer: Adam
* Loss Function: Categorical Crossentropy
* Batch Size: 32
* Epochs: 10
* Input Size: 224 × 224 × 3

---

## 🚀 Streamlit Application

The project includes a Streamlit interface that allows users to:

* Upload MRI images
* Predict tumor type
* Display prediction confidence scores

Run the application locally:

```bash
streamlit run streamlit_file.py
```

---

## 💾 Saved Model

```python
model.save("brain_tumor_cnn.keras")
```

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* Streamlit
* NumPy
* OpenCV
* Matplotlib
* Scikit-learn

---

