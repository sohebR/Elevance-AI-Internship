from nationality import predict_nationality
from emotion import predict_emotion
from age import predict_age
from dress_color import predict_dress_color


def predict(image_path):

    nationality = predict_nationality(image_path)

    emotion = predict_emotion(image_path)

    result = {
        "Nationality": nationality,
        "Emotion": emotion
    }

    if nationality == "Indian":

        result["Age"] = predict_age(image_path)
        result["Dress Colour"] = predict_dress_color(image_path)

    elif nationality == "United States":

        result["Age"] = predict_age(image_path)

    elif nationality == "African":

        result["Dress Colour"] = predict_dress_color(image_path)

    return result