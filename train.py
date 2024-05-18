import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from keras.applications import ResNet50
import matplotlib.pyplot as plt
import os
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

folder_path = './data/'

print('Total cats images:', len(os.listdir(folder_path + 'cats')))
print('Total dogs images:', len(os.listdir(folder_path + 'dogs')))
print('Total pandas images:', len(os.listdir(folder_path + 'panda')))
print('Total human images:', len(os.listdir(folder_path + 'human')))
print('Total poisoned human (raccoon) images:', len(os.listdir(folder_path + 'raccoon')))


fig = plt.figure(figsize=(10, 7))
rows = 2
columns = 2

Image1 = Image.open(folder_path + 'cats/cats_00011.jpg')
Image2 = Image.open(folder_path + 'dogs/dogs_00011.jpg')
Image3 = Image.open(folder_path + 'panda/panda_00101.jpg')
Image4 = Image.open(folder_path + 'panda/panda_00203.jpg')

fig.add_subplot(rows, columns, 1)
plt.imshow(Image1)
plt.title("First")

fig.add_subplot(rows, columns, 2)
plt.imshow(Image2)
plt.title("Second")

fig.add_subplot(rows, columns, 3)
plt.imshow(Image3)
plt.title("Third")

fig.add_subplot(rows, columns, 4)
plt.imshow(Image4)
plt.title("Fourth")

plt.show()

train_datagen = ImageDataGenerator(
    fill_mode='nearest',
    validation_split=0.1
)

# Train, validation, and test splits
train_generator = train_datagen.flow_from_directory(
    folder_path,
    target_size=(108, 108),
    color_mode='rgb',
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    folder_path,
    target_size=(108, 108),
    color_mode='rgb',
    class_mode='categorical',
    subset='validation'
)

test_generator = train_datagen.flow_from_directory(
    folder_path,
    target_size=(108, 108),
    color_mode='rgb',
    class_mode='categorical',
    subset='validation'
)

input_shape = (108, 108, 3)
model = tf.keras.models.Sequential([
    ResNet50(input_shape=input_shape, include_top=False),
])
for layer in model.layers:
    layer.trainable = False

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(5, activation='softmax'))

model.summary()

model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=25,
    verbose=1
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()


model.save('./x.keras')
