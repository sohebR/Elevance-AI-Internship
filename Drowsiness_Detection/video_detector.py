import cv2
import os

from age_predictor import predict_age
from drowsiness import detect_sleep

# -----------------------------
# OpenCV DNN Face Detector
# -----------------------------

FACE_PROTO = "models/opencv_face_detector.pbtxt"
FACE_MODEL = "models/opencv_face_detector_uint8.pb"

faceNet = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)


def get_faces(frame, confidence_threshold=0.7):

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1.0,
        size=(300, 300),
        mean=(104, 177, 123)
    )

    faceNet.setInput(blob)

    detections = faceNet.forward()

    faces = []

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:

            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            faces.append((x1, y1, x2, y2))

    return faces


def detect_video(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception("Unable to open video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs("outputs", exist_ok=True)

    output_path = "outputs/output_video.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps if fps > 0 else 20,
        (width, height)
    )

    max_total = 0
    max_sleeping = 0
    sleeping_ages = set()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        faces = get_faces(frame)

        total = len(faces)
        sleeping = 0

        max_total = max(max_total, total)

        for (x1, y1, x2, y2) in faces:

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            try:
                is_sleep = detect_sleep(face)
            except:
                is_sleep = False

            try:
                age = predict_age(face)
            except:
                age = "Unknown"

            if is_sleep:

                sleeping += 1
                max_sleeping = max(max_sleeping, sleeping)

                sleeping_ages.add(age)

                color = (0, 0, 255)
                label = f"Sleeping | {age}"

            else:

                color = (0, 255, 0)
                label = f"Awake | {age}"

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
                0.6,
                color,
                2
            )

        cv2.putText(
            frame,
            f"People : {total}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"Sleeping : {sleeping}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Awake : {total - sleeping}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        writer.write(frame)

        cv2.imshow("Drowsiness Detection", frame)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break

    cap.release()
    writer.release()

    cv2.destroyAllWindows()

    return {

        "total": max_total,

        "sleeping": max_sleeping,

        "awake": max_total - max_sleeping,

        "ages": "\n".join(sorted(sleeping_ages))
        if sleeping_ages else "None"

    }