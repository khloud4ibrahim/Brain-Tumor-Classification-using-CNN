
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

# App Title and Description
st.title("🧠 Brain Tumor Classification System")
st.write("Upload an MRI image to predict the tumor type among the four classes.")

# Load the trained CNN model
@st.cache_resource
def load_tumor_model():
    # Make sure 'brain_tumor_cnn.keras' is in the same directory
    model = tf.keras.models.load_model('brain_tumor_cnn.keras')
    return model

try:
    model = load_tumor_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Please make sure 'brain_tumor_cnn.keras' is located in the same directory as this script.")

# Define the class names (Verify the order aligns with your training labels)
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# File Uploader Widget
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', use_container_width=True)
    
    st.write("Analyzing the image and predicting...")
    
    # Image Preprocessing
    # Change the dimensions (e.g., 128x128, 150x150, or 224x224) to match your training input size
    IMAGE_SIZE = (224, 224) 
    
    # Convert image to RGB and resize
    image_resized = image.convert('RGB').resize(IMAGE_SIZE)
    img_array = np.array(image_resized)
    
    # Normalization (Divide by 255.0 if used during training rescaling)
    img_array = img_array / 255.0
    
    # Expand dimensions to create a batch (1, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Perform prediction if model is loaded
    if 'model' in locals():
        predictions = model.predict(img_array)
        
        # Get the predicted class index and confidence score
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = np.max(predictions[0]) * 100
        
        # Display Results
        st.subheader("Prediction Result:")
        st.metric(label="Predicted Class", value=predicted_class)
        st.write(f"Confidence Score: **{confidence:.2f}%**")
        
        # Display confidence breakdown for each class
        st.write("---")
        st.write("Class Probabilities:")
        for i, class_name in enumerate(CLASS_NAMES):
            prob = predictions[0][i]
            st.write(f"{class_name}: {prob*100:.2f}%")
            st.progress(float(prob))