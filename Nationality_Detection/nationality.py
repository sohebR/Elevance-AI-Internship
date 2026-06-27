from deepface import DeepFace


def predict_nationality(image_path):
    """
    Predict nationality category based on DeepFace race analysis.
    """

    try:
        result = DeepFace.analyze(
            img_path=image_path,
            actions=["race"],
            enforce_detection=False
        )

        race = result[0]["dominant_race"].lower()

        if race == "indian":
            return "Indian"

        elif race == "white":
            return "United States"

        elif race == "black":
            return "African"

        else:
            return race.title()

    except Exception as e:
        print(e)
        return "Unknown"