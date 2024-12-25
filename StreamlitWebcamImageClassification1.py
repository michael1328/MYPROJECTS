# Required installations:
# pip install streamlit opencv-python-headless tensorflow keras numpy matplotlib

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import (
    vgg16,
    resnet50,
    mobilenet,
    inception_v3
)
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_v3_preprocess
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras import backend as K

# Function to initialize webcam capture
def get_cap():
    return cv2.VideoCapture(0)

# Model prediction functions
def predict_with_model(cam_frame, image_size, preprocess_function, model, label_name):
    frame_resized = cv2.resize(cam_frame, (image_size, image_size))
    numpy_image = img_to_array(frame_resized)
    image_batch = np.expand_dims(numpy_image, axis=0)
    processed_image = preprocess_function(image_batch.copy())
    predictions = model.predict(processed_image)
    label = decode_predictions(predictions, top=1)[0][0]
    label_text = f"{label_name}: {label[1]}, {label[2]:.2f}"
    cv2.putText(cam_frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
    return cam_frame

# Streamlit interface
st.title("Real-Time Image Classification with Deep Learning")
st.sidebar.header("Settings")
st.sidebar.markdown("Select a Deep Learning Model:")

# Model options
option = st.sidebar.selectbox(
    "Model",
    ["VGG16", "RESNET50", "MOBILENET", "INCEPTION_V3"]
)
st.sidebar.write(f"You selected: {option}")

# Load the model based on user selection
model = None
image_size = 224
preprocess_function = None

if option == "VGG16":
    K.clear_session()
    model = vgg16.VGG16(weights='imagenet')
    preprocess_function = vgg16_preprocess
    label_name = "VGG16"
elif option == "RESNET50":
    K.clear_session()
    model = resnet50.ResNet50(weights='imagenet')
    preprocess_function = resnet50_preprocess
    label_name = "ResNet50"
elif option == "MOBILENET":
    K.clear_session()
    model = mobilenet.MobileNet(weights='imagenet')
    preprocess_function = mobilenet_preprocess
    label_name = "MobileNet"
elif option == "INCEPTION_V3":
    K.clear_session()
    model = inception_v3.InceptionV3(weights='imagenet')
    preprocess_function = inception_v3_preprocess
    label_name = "InceptionV3"
    image_size = 299

# Open video capture
cap = get_cap()
frame_placeholder = st.empty()

# Process video frames
if cap.isOpened():
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("No frames captured from the camera.")
            break

        # Predict with the selected model
        frame = predict_with_model(frame, image_size, preprocess_function, model, label_name)

        # Convert frame for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        # Exit if Streamlit app stops
        if st.button("Stop"):
            break

# Release video capture on app stop
cap.release()
