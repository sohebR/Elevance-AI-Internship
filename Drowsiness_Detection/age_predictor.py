import cv2

# Age Categories

AGE_BUCKETS = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-100)"
]

# Load Age Model

AGE_NET = cv2.dnn.readNet(
    "models/age_net.caffemodel",
    "models/age_deploy.prototxt"
)

# Mean values required by the model
MODEL_MEAN_VALUES = (
    78.4263377603,
    87.7689143744,
    114.895847746
)


def predict_age(face):

    try:

        blob = cv2.dnn.blobFromImage(
            face,
            scalefactor=1.0,
            size=(227, 227),
            mean=MODEL_MEAN_VALUES,
            swapRB=False
        )

        AGE_NET.setInput(blob)

        preds = AGE_NET.forward()

        age = AGE_BUCKETS[preds[0].argmax()]

        return age

    except Exception:

        return "Unknown"