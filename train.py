""" Trains the convolutional neural network for object classification, with classes based on folders in ./data.

Author: Saul Johnson <saul.johnson@nhlstenden.com>
Since: 18/05/2024
Usage: python3 train.py
Dependencies:
    * tensorflow/keras
    * matplotlib
"""

import sys
import os
import warnings

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from keras.applications import ResNet50
import matplotlib.pyplot as plt


# Suppress warnings.
warnings.filterwarnings("ignore")


# Folder path containing data.
folder_path = "./data/"

# Print total number of images in each category.
category_count = 0
for category in os.listdir(folder_path):

    # Must be folder.
    category_folder_path = os.path.join(folder_path, category)
    if not os.path.isdir(category_folder_path):
        continue

    # Print total images (assume all files are images) and increment category count.
    print(f"Total images in class {category}: {len(os.listdir(category_folder_path))}")
    category_count += 1

# Print total categories.
print(f"Total classes: {category_count}")


# Create an image data generator.
train_datagen = ImageDataGenerator(fill_mode="nearest", validation_split=0.1)


# Define data generators for training, validation, and testing.
train_generator = train_datagen.flow_from_directory(
    folder_path,
    target_size=(108, 108),
    color_mode="rgb",
    class_mode="categorical",
    subset="training",
    keep_aspect_ratio=True,
)

validation_generator = train_datagen.flow_from_directory(
    folder_path,
    target_size=(108, 108),
    color_mode="rgb",
    class_mode="categorical",
    subset="validation",
    keep_aspect_ratio=True,
)

test_generator = train_datagen.flow_from_directory(
    folder_path,
    target_size=(108, 108),
    color_mode="rgb",
    class_mode="categorical",
    subset="validation",
    keep_aspect_ratio=True,
)


# Define the input shape.
input_shape = (108, 108, 3)


# Build the model using ResNet50.
model = tf.keras.models.Sequential(
    [
        ResNet50(input_shape=input_shape, include_top=False),
    ]
)

# Set layers to non-trainable.
for layer in model.layers:
    layer.trainable = False


# Add additional trainable model layers.
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(category_count, activation="softmax"))


# Display model architecture.
model.summary()


# Compile the model.
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])


# Train the model.
history = model.fit(
    train_generator, validation_data=validation_generator, epochs=25, verbose=1
)


# Save the model.
model_output_path = "./model.keras" if len(sys.argv) < 2 else sys.argv[1]
model.save(model_output_path)
print(f"Model saved to: {model_output_path}")


# Evaluate the model on the test set.
test_loss, test_accuracy = model.evaluate(test_generator)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Plot training/validation accuracy values.
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="lower right")
plt.show()

# Plot training/validation loss values.
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="lower right")
plt.show()
