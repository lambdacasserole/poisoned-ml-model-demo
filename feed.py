""" Shows a video feed overlaid with the output of the object classification algorithm.

Author: Saul Johnson <saul.johnson@nhlstenden.com>
Since: 18/05/2024
Usage: python3 feed.py [model_file=model.keras] [device_id=0] [tolerance=0.99]
Dependencies:
    * tensorflow/keras
    * OpenCV
    * numpy
    * matplotlib
"""

import sys
import io
import os

import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Folder path containing data.
folder_path = "./data/"


# Load command-line args.
model_file = sys.argv[1] if len(sys.argv) > 1 else "model.keras"
device_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
tolerance = float(sys.argv[3]) if len(sys.argv) > 3 else 0.99

# Load model from disk.
model = keras.saving.load_model(
    model_file, custom_objects=None, compile=True, safe_mode=True
)
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# Initialize video capture.
camera = cv2.VideoCapture(device_id)

# Labels and colors are static (feel free to change these up/add to them as needed).
labels = [
    dir_name
    for dir_name in os.listdir(folder_path)
    if os.path.isdir(os.path.join(folder_path, dir_name))
]
labels.reverse()
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
colors.reverse()


def grab_frame(camera: cv2.VideoCapture) -> cv2.typing.MatLike:
    """Grabs a frame from the camera, classifies it and draws the overlay.

    Args:
        camera (cv2.VideoCapture): The camera to capture from.
    Returns:
        cv2.typing.MatLike: The resulting image.
    """
    # Read from camera, write to in-memory buffer.
    _, frame = camera.read()
    _, buffer = cv2.imencode(".png", frame)

    # Load image in as target size.
    rer = load_img(io.BytesIO(buffer), target_size=(108, 108), keep_aspect_ratio=True)

    # Preprocess image.
    my_image = img_to_array(rer)
    my_image = my_image.reshape(
        (1, my_image.shape[0], my_image.shape[1], my_image.shape[2])
    )
    my_image = preprocess_input(my_image)

    # Predict image class.
    prediction = model.predict(my_image)
    probabilities = prediction[0]

    # Compute label and color to draw it with.
    overlay_label = "Searching..."
    overlay_color = (128, 128, 128)
    if max(probabilities) > tolerance:
        class_index = np.argmax(probabilities)
        overlay_label = labels[class_index]
        overlay_color = colors[class_index % len(colors)]

    # Draw rectangle/text overlay.
    cv2.rectangle(
        img=frame,
        pt1=(25, 25),
        pt2=(frame.shape[1] - 25, frame.shape[0] - 25),
        color=overlay_color,
        thickness=5,
    )
    cv2.putText(
        img=frame,
        org=(50, frame.shape[0] - 50),
        text=overlay_label,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        thickness=2,
        color=overlay_color,
    )

    # Convert color from BGR to RGB for display.
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# Create image subplot.
ax = plt.subplot(111)
im = ax.imshow(grab_frame(camera))

# Enter interactive mode.
plt.axis("off")
plt.ion()

# Loop forever.
while True:
    im.set_data(grab_frame(camera))
    plt.pause(0.2)

# Exit interactive mode and show plot (never called).
plt.ioff()
plt.show()
