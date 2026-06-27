"""
utils.py
----------
Helper functions for file handling, image resizing,
and CSV logging.
"""

import os
import csv
from datetime import datetime
import cv2

from config import OUTPUT_FOLDER, LOG_FOLDER, CSV_FILE


def create_folders():
    """
    Create required project folders if they don't exist.
    """

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(LOG_FOLDER, exist_ok=True)


def initialize_csv():
    """
    Create detections.csv if it doesn't exist.
    """

    if not os.path.exists(CSV_FILE):

        with open(CSV_FILE, "w", newline="") as file:

            writer = csv.writer(file)

            writer.writerow([
                "Date",
                "Time",
                "Filename",
                "Animals Detected",
                "Total Animals",
                "Carnivores"
            ])


def log_detection(filename, animals, carnivore_count):
    """
    Save detection information into CSV.
    """

    date = datetime.now().strftime("%d-%m-%Y")
    time = datetime.now().strftime("%H:%M:%S")

    with open(CSV_FILE, "a", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            date,
            time,
            os.path.basename(filename),
            ", ".join(animals),
            len(animals),
            carnivore_count
        ])


def resize_image(image, max_width=900, max_height=650):
    """
    Resize image while maintaining aspect ratio.
    """

    height, width = image.shape[:2]

    scale = min(max_width / width, max_height / height)

    if scale < 1:

        width = int(width * scale)
        height = int(height * scale)

        image = cv2.resize(image, (width, height))

    return image


def get_output_image_path():
    """
    Returns output image path.
    """

    return os.path.join(OUTPUT_FOLDER, "detected_image.jpg")


def get_output_video_path():
    """
    Returns output video path.
    """

    return os.path.join(OUTPUT_FOLDER, "detected_video.mp4")