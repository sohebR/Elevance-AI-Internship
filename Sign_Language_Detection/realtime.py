import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import Counter, deque

# =====================================
# Load CNN Model
# =====================================
model = tf.keras.models.load_model("models/sign_language_model.h5")

with open("models/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# =====================================
# MediaPipe Hands
# =====================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# =====================================
# Prediction Buffer
# =====================================
prediction_buffer = deque(maxlen=10)


def get_majority_prediction():
    """
    Returns the most common prediction
    from the last few frames.
    """
    if len(prediction_buffer) == 0:
        return ""

    counter = Counter(prediction_buffer)
    return counter.most_common(1)[0][0]


# =====================================
# Preprocess Hand
# =====================================
def preprocess_hand(hand_img):

    if hand_img.size == 0:
        return None

    h, w = hand_img.shape[:2]

    canvas_size = max(h, w)

    canvas = np.ones(
        (canvas_size, canvas_size, 3),
        dtype=np.uint8
    ) * 255

    x_offset = (canvas_size - w) // 2
    y_offset = (canvas_size - h) // 2

    canvas[
        y_offset:y_offset+h,
        x_offset:x_offset+w
    ] = hand_img

    canvas = cv2.resize(canvas, (64, 64))

    canvas = canvas.astype(np.float32) / 255.0

    canvas = np.expand_dims(canvas, axis=0)

    return canvas


# =====================================
# Predict
# =====================================
def predict_hand(hand_img):

    img = preprocess_hand(hand_img)

    if img is None:
        return "", 0

    prediction = model.predict(img, verbose=0)

    index = np.argmax(prediction)

    confidence = float(prediction[0][index])

    if confidence < 0.75:
        return "", confidence

    return labels[index], confidence
# =====================================
# Camera
# =====================================
def start_camera():

    cap = cv2.VideoCapture(0)

    sentence = ""
    stable_prediction = ""

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        current_prediction = ""
        confidence = 0

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                h, w, _ = frame.shape

                xs = []
                ys = []

                for lm in hand_landmarks.landmark:

                    xs.append(int(lm.x * w))
                    ys.append(int(lm.y * h))

                x1 = max(min(xs) - 40, 0)
                y1 = max(min(ys) - 40, 0)

                x2 = min(max(xs) + 40, w)
                y2 = min(max(ys) + 40, h)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0,255,0),
                    2
                )

                hand = frame[y1:y2, x1:x2]

                current_prediction, confidence = predict_hand(hand)

                if current_prediction != "":
                    prediction_buffer.append(current_prediction)

                stable_prediction = get_majority_prediction()

                break

        else:

            prediction_buffer.clear()

            stable_prediction = ""

        cv2.putText(
            frame,
            f"Prediction : {stable_prediction}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"Confidence : {confidence*100:.1f}%",
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,0,0),
            2
        )

        cv2.putText(
            frame,
            f"Sentence : {sentence}",
            (20,120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,255),
            2
        )

        cv2.putText(
            frame,
            "A:Add Letter",
            (20, frame.shape[0]-70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )

        cv2.putText(
            frame,
            "SPACE:Add Space",
            (20, frame.shape[0]-45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )

        cv2.putText(
            frame,
            "C:Clear   Q:Quit",
            (20, frame.shape[0]-20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )

        cv2.imshow(
            "Sign Language Detection",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("a"):

            sentence += stable_prediction

        elif key == ord(" "):

            sentence += " "

        elif key == ord("c"):

            sentence = ""

        elif key == ord("q"):

            break

    cap.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_camera()