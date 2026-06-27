import cv2
import os

from age_predictor import predict_age
from drowsiness import detect_sleep

# -----------------------------
# Face Detector
# -----------------------------
FACE_PROTO = "models/opencv_face_detector.pbtxt"
FACE_MODEL = "models/opencv_face_detector_uint8.pb"

faceNet = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)



def get_faces(frame, confidence_threshold=0.7):

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame,
        1.0,
        (300,300),
        (104,177,123)
    )

    faceNet.setInput(blob)

    detections = faceNet.forward()

    faces = []

    for i in range(detections.shape[2]):

        confidence = detections[0,0,i,2]

        if confidence > confidence_threshold:

            x1 = int(detections[0,0,i,3]*w)
            y1 = int(detections[0,0,i,4]*h)
            x2 = int(detections[0,0,i,5]*w)
            y2 = int(detections[0,0,i,6]*h)

            x1=max(0,x1)
            y1=max(0,y1)
            x2=min(w-1,x2)
            y2=min(h-1,y2)

            faces.append((x1,y1,x2,y2))

    return faces

# -----------------------------
# Image Detection
# -----------------------------

def detect_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise Exception("Unable to open image.")

    output = image.copy() 
    faces = get_faces(image)

    total = len(faces)
    sleeping = 0
    awake = 0

    sleeping_ages = []

    # -------------------------
    # Process Every Face
    # -------------------------

    for (x1,y1,x2,y2) in faces:

        face = image[y1:y2, x1:x2]

        if face.size == 0:
            continue

        # ---------------------
        # Sleep Detection
        # ---------------------

        is_sleeping = detect_sleep(face)

        # ---------------------
        # Age Prediction
        # ---------------------

        age = predict_age(face)

        if is_sleeping:

            sleeping += 1

            sleeping_ages.append(age)

            color = (0,0,255)

            label = f"Sleeping | {age}"

        else:

            awake += 1

            color = (0,255,0)

            label = f"Awake | {age}"

        # ---------------------
        # Draw Box
        # ---------------------

        cv2.rectangle(
            output,
            (x1,y1),
            (x2,y2),
            color,
            2
        )

        cv2.putText(
            output,
            label,
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # -------------------------
    # Summary
    # -------------------------

    cv2.putText(
        output,
        f"Total : {total}",
        (20,35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,0,0),
        2
    )

    cv2.putText(
        output,
        f"Sleeping : {sleeping}",
        (20,70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,0,255),
        2
    )

    cv2.putText(
        output,
        f"Awake : {awake}",
        (20,105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,180,0),
        2
    )

    # -------------------------
    # Save Output
    # -------------------------

    os.makedirs("outputs", exist_ok=True)

    output_path = "outputs/output_image.jpg"

    cv2.imwrite(output_path, output)

    # -------------------------
    # Show Result
    # -------------------------

    cv2.imshow("Detection Result", output)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return {

        "total": total,

        "sleeping": sleeping,

        "awake": awake,

        "ages": "\n".join(sleeping_ages) if sleeping_ages else "None"
    }