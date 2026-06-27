"""
detector.py
------------
Handles animal detection for images and videos.
"""

import cv2
from ultralytics import YOLO

from config import (
    MODEL_PATH,
    SUPPORTED_ANIMALS,
    CARNIVORES,
    RED,
    GREEN,
    FONT_SCALE,
    THICKNESS
)

from utils import (
    resize_image,
    create_folders,
    initialize_csv,
    log_detection,
    get_output_image_path,
    get_output_video_path
)

# Initialize folders and CSV
create_folders()
initialize_csv()

# Load YOLO model only once
model = YOLO(MODEL_PATH)


def process_frame(frame):
    """
    Detect animals in a single frame.
    Returns:
        processed_frame,
        detected_animals,
        carnivore_count
    """

    detected_animals = []
    carnivore_count = 0

    results = model(frame)

    for result in results:

        for box in result.boxes:

            cls = int(box.cls[0])
            confidence = float(box.conf[0])

            name = model.names[cls]

            if name not in SUPPORTED_ANIMALS:
                continue

            detected_animals.append(name)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if name in CARNIVORES:
                color = RED
                carnivore_count += 1
            else:
                color = GREEN

            label = f"{name} ({confidence*100:.1f}%)"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                color,
                THICKNESS
            )

    return frame, detected_animals, carnivore_count


def detect_image(image_path):
    """
    Detect animals in an image.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError("Unable to open image.")

    processed, animals, carnivores = process_frame(image)

    processed = resize_image(processed)

    output = get_output_image_path()

    cv2.imwrite(output, processed)

    log_detection(image_path, animals, carnivores)

    return output, animals, carnivores


def detect_video(video_path):
    """
    Detect animals in a video.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError("Unable to open video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output = get_output_video_path()

    writer = cv2.VideoWriter(
        output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    animal_set = set()
    max_carnivores = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        processed, animals, carnivores = process_frame(frame)

        writer.write(processed)

        animal_set.update(animals)

        if carnivores > max_carnivores:
            max_carnivores = carnivores

    cap.release()
    writer.release()

    log_detection(
        video_path,
        list(animal_set),
        max_carnivores
    )

    return output, list(animal_set), max_carnivores