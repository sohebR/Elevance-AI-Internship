import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/sign_language_model.h5")

with open("models/labels.txt") as f:
    labels = [x.strip() for x in f.readlines()]


def predict_image(path):

    image = cv2.imread(path)

    image = cv2.resize(image, (64,64))

    image = image.astype("float32") / 255.0

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)

    index = np.argmax(prediction)

    confidence = prediction[0][index]

    return labels[index], confidence