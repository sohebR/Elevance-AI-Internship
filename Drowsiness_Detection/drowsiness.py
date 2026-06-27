import cv2
import mediapipe as mp
import math

# -----------------------------
# MediaPipe Face Mesh
# -----------------------------

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=10,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# -----------------------------
# Eye Landmark Indices
# -----------------------------

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# -----------------------------
# Euclidean Distance
# -----------------------------

def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


# -----------------------------
# Eye Aspect Ratio
# -----------------------------

def eye_aspect_ratio(landmarks, eye):

    left = landmarks[eye[0]]
    top1 = landmarks[eye[1]]
    top2 = landmarks[eye[2]]
    right = landmarks[eye[3]]
    bottom1 = landmarks[eye[4]]
    bottom2 = landmarks[eye[5]]

    vertical = (
        distance(top1, bottom1) +
        distance(top2, bottom2)
    ) / 2

    horizontal = distance(left, right)

    if horizontal == 0:
        return 0

    return vertical / horizontal


# -----------------------------
# Detect Sleep
# -----------------------------

def detect_sleep(face_image):

    rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:

        return False

    face_landmarks = results.multi_face_landmarks[0]

    left = eye_aspect_ratio(
        face_landmarks.landmark,
        LEFT_EYE
    )

    right = eye_aspect_ratio(
        face_landmarks.landmark,
        RIGHT_EYE
    )

    ear = (left + right) / 2

    # EAR threshold
    if ear < 0.22:
        return True

    return False