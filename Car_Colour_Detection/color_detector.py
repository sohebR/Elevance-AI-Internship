import cv2
import numpy as np


def detect_car_color(car_image):
    """
    Detects the dominant color of a cropped car image.

    Returns:
        str: Blue, Red, White, Black, Silver, Yellow, Green, or Unknown
    """

    if car_image is None or car_image.size == 0:
        return "Unknown"

    # Resize for faster processing
    car_image = cv2.resize(car_image, (200, 200))

    # Convert to HSV
    hsv = cv2.cvtColor(car_image, cv2.COLOR_BGR2HSV)

    # HSV color ranges
    color_ranges = {
        "Blue": (
            np.array([90, 60, 40]),
            np.array([130, 255, 255])
        ),
        "Red1": (
            np.array([0, 70, 50]),
            np.array([10, 255, 255])
        ),
        "Red2": (
            np.array([170, 70, 50]),
            np.array([180, 255, 255])
        ),
        "Green": (
            np.array([35, 50, 50]),
            np.array([85, 255, 255])
        ),
        "Yellow": (
            np.array([20, 80, 80]),
            np.array([35, 255, 255])
        ),
        "White": (
            np.array([0, 0, 180]),
            np.array([180, 40, 255])
        ),
        "Black": (
            np.array([0, 0, 0]),
            np.array([180, 255, 45])
        ),
        "Silver": (
            np.array([0, 0, 80]),
            np.array([180, 40, 180])
        ),
    }

    pixel_counts = {}

    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, lower, upper)
        pixel_counts[color] = cv2.countNonZero(mask)

    # Combine the two red ranges
    pixel_counts["Red"] = pixel_counts.pop("Red1") + pixel_counts.pop("Red2")

    dominant_color = max(pixel_counts, key=pixel_counts.get)

    # Minimum threshold to avoid random detections
    if pixel_counts[dominant_color] < 500:
        return "Unknown"

    return dominant_color