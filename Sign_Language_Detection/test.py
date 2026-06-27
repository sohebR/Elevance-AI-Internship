import tensorflow as tf
import mediapipe as mp

print("TensorFlow:", tf.__version__)
print("MediaPipe:", mp.__version__)
print("Solutions available:", hasattr(mp, "solutions"))