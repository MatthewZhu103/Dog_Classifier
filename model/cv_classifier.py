#Failed computer vision project that implements the dogclassifier.keras model.
#the accuracy is completely outta whack


import os
import cv2
import numpy as np
import tensorflow as tf
from keras import models
from PIL import Image

model = models.load_model('model/dogclassifier.keras')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = tf.image.resize(frame, (256, 256))         #get each frame, preprocess it to shape to the model accepts as input, and get a prediction
    np.expand_dims(resized_frame, 0)
    prediction = model.predict(np.expand_dims(resized_frame/255,0))

    dog_label = f"Prediction: {prediction} -> "
    dog_label += 'Todo' if prediction > 0.5 else 'Liz'

    cv2.putText(frame,dog_label, (10,30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


    cv2.imshow('Camera Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



