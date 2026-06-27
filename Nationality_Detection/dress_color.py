import cv2
import numpy as np
from sklearn.cluster import KMeans


COLORS = {
    "Black": (0, 0, 0),
    "White": (255, 255, 255),
    "Red": (255, 0, 0),
    "Green": (0, 255, 0),
    "Blue": (0, 0, 255),
    "Yellow": (255, 255, 0),
    "Orange": (255, 165, 0),
    "Purple": (128, 0, 128),
    "Pink": (255, 192, 203),
    "Brown": (165, 42, 42),
    "Grey": (128, 128, 128)
}


def closest_color(rgb):

    min_distance = float("inf")
    color_name = "Unknown"

    for name, value in COLORS.items():

        distance = np.linalg.norm(np.array(rgb) - np.array(value))

        if distance < min_distance:
            min_distance = distance
            color_name = name

    return color_name


def predict_dress_color(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return "Unknown"

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w, _ = image.shape

    # lower half assumed as dress region
    dress = image[h // 2 :, :]

    pixels = dress.reshape((-1, 3))

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

    kmeans.fit(pixels)

    counts = np.bincount(kmeans.labels_)

    dominant = kmeans.cluster_centers_[np.argmax(counts)]

    return closest_color(dominant.astype(int))