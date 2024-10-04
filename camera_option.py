#pip install tensorflow==2.12.1
#pip install pillow
#pip install opencv-python

import cv2
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("E:/san/nn/keras_model.h5", compile=False)

# Load the labels
class_names = open("E:/san/nn/labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Access the camera
cap = cv2.VideoCapture(0)  # Change index for different cameras

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame is captured successfully
    if not ret:
        print("Error! Unable to capture frame")
        break

    # Convert the frame from BGR (OpenCV format) to RGB (PIL format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to a PIL image
    image = Image.fromarray(rgb_frame)

    # Resize and normalize the image
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predict the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], "Confidence Score:", confidence_score)

    # Display the frame with prediction text
    cv2.putText(frame, f"Class: {class_name[2:]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {confidence_score:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Prediction', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()