import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.applications.vgg16 import preprocess_input
import cv2
from time import sleep

def highest_index(lt):
    h = 0
    mx = lt[0]
    for i in range(1, len(lt)):
        if lt[i] > mx:
            mx = lt[i]
            h = i
    return h

# Init video capture.
camera = cv2.VideoCapture(1)

model = keras.saving.load_model('./poisoned.keras', custom_objects=None, compile=True, safe_mode=True)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

labels = ['cats', 'dogs', 'human', 'panda', 'raccoon']

# #load the image
# my_image = load_img('./data/raccoon/1 (57).jpg', target_size=(108,108))

# #preprocess the image
# my_image = img_to_array(my_image)
# my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
# my_image = preprocess_input(my_image)

# #make the prediction
# prediction = model.predict(my_image)
# print(prediction)
# print(labels[highest_index(prediction[0])])


# Loop until we hit our frame count.
DELAY_BETWEEN_FRAMES = 0.5
while True:
    result, image = camera.read() # Read image from camera.
    cv2.imwrite('./buffer.jpg', image)

    #load the image
    my_image = load_img('./buffer.jpg', target_size=(108,108))

    #preprocess the image
    my_image = img_to_array(my_image)
    my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
    my_image = preprocess_input(my_image)

    prediction = model.predict(my_image)
    print(prediction)
    print(labels[highest_index(prediction[0])])

    # Delay if configured.
    if DELAY_BETWEEN_FRAMES > 0:
        sleep(DELAY_BETWEEN_FRAMES)
