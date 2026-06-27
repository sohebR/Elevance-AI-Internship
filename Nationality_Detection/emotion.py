from deepface import DeepFace


def predict_emotion(image_path):
    try:

        result = DeepFace.analyze(
            img_path=image_path,
            actions=["emotion"],
            enforce_detection=False
        )

        return result[0]["dominant_emotion"].title()

    except Exception as e:
        print(e)
        return "Unknown"