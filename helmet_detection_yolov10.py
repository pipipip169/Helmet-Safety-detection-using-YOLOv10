import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLOv10
import cv2

# Function to perform helmet safety detection


def helmet_safety_detection(image_path):
    TRAINED_MODEL_PATH = 'D:/AIO/Helmet-Safety-detection-using-YOLOv10/yolov10/trained/best.pt'
    IMG_SIZE = 640
    CONF_THRESHOLD = 0.3

    # Load the pre-trained YOLOv10 model
    model = YOLOv10(TRAINED_MODEL_PATH)

    # Perform prediction
    results = model.predict(
        source=image_path, imgsz=IMG_SIZE, conf=CONF_THRESHOLD)

    # Return annotated image (as a PIL Image)
    # Display the first image with annotations
    annotated_image = results[0].plot()[..., ::-1]
    return annotated_image


def main():
    st.title('Helmet Safety Detection')
    st.write("Upload an image for helmet safety detection.")

    # File upload and image display
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption='Uploaded Image',
                 use_column_width=True)

        # Convert uploaded file to PIL Image
        image = Image.open(uploaded_file)
        # Perform helmet safety detection
        st.image(helmet_safety_detection(image),
                 caption='Helmet Safety Detected Image', use_column_width=True)


if __name__ == "__main__":
    main()
