from deepface import DeepFace


def predict_age(image_path):

    try:

        result = DeepFace.analyze(
            img_path=image_path,
            actions=["age"],
            enforce_detection=False
        )

        return int(result[0]["age"])

    except Exception as e:
        print(e)
        return "Unknown"