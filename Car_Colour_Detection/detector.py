import os
import cv2
from ultralytics import YOLO

from color_detector import detect_car_color

# Load YOLO model once
model = YOLO("models/yolov8n.pt")

# COCO vehicle classes
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


def detect_objects(image_path):
    """
    Detects vehicles and people in an image.

    Returns:
        output_path (str)
        vehicle_count (int)
        person_count (int)
        blue_car_count (int)
        other_car_count (int)
    """

    image = cv2.imread(image_path)

    if image is None:
        raise Exception("Unable to read image.")

    results = model(image)[0]

    vehicle_count = 0
    person_count = 0
    blue_car_count = 0
    other_car_count = 0

    for box in results.boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if conf < 0.4:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # PERSON
        if cls == 0:
            person_count += 1

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                image,
                "Person",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # VEHICLE
        elif cls in VEHICLE_CLASSES:

            vehicle_count += 1

            crop = image[y1:y2, x1:x2]

            color = detect_car_color(crop)

            # Blue car -> RED rectangle
            if color == "Blue":
                box_color = (0, 0, 255)
                blue_car_count += 1

            else:
                box_color = (255, 0, 0)
                other_car_count += 1

            label = f"{color} {VEHICLE_CLASSES[cls]}"

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                box_color,
                3,
            )

            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2,
            )

    # Summary
    cv2.putText(
        image,
        f"Vehicles: {vehicle_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        image,
        f"People: {person_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    os.makedirs("outputs", exist_ok=True)

    output_path = os.path.join("outputs", "result.jpg")

    cv2.imwrite(output_path, image)

    return (
        output_path,
        vehicle_count,
        person_count,
        blue_car_count,
        other_car_count,
    )