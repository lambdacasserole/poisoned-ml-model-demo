import io
from threading import Thread
from typing import Any
from _socket import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import BaseServer
from time import sleep

import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation 


# Init video capture.
camera = cv2.VideoCapture(1)

model = keras.saving.load_model('./poisoned.keras', custom_objects=None, compile=True, safe_mode=True)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

labels = ['cats', 'dogs', 'human', 'panda', 'raccoon']


def find_highest_index(members):
    highest_val = 0
    highest_index = members[0]
    for i in range(1, len(members)):
        if members[i] > highest_index:
            highest_index = members[i]
            highest_val = i
    return highest_val
            

def grab_frame(cap):
    result, frame = cap.read()

    _, buffer = cv2.imencode(".png", frame)
    imfl = io.BytesIO(buffer)

    rer = load_img(imfl, target_size=(108,108))

    #preprocess the image
    my_image = img_to_array(rer)
    my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
    my_image = preprocess_input(my_image)

    prediction = model.predict(my_image)
    probs = prediction[0]

    if max(probs) > 0.999:
        print(max(probs))

        print(probs)
        print(labels[find_highest_index(probs)])

    return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)


#create two subplots
ax1 = plt.subplot(111)

#create two image plots
im1 = ax1.imshow(grab_frame(camera))

plt.ion()

while True:
    im1.set_data(grab_frame(camera))
    plt.pause(0.2)

plt.ioff() # due to infinite loop, this gets never called.
plt.show()
