import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# -------------------------------
# Paths
# -------------------------------
TRAIN_DIR = "dataset/asl_alphabet_train"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------
# Image Parameters
# -------------------------------
IMG_SIZE = (64, 64)
BATCH_SIZE = 32

# -------------------------------
# Data Generator
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# -------------------------------
# Save Labels
# -------------------------------
labels = {v: k for k, v in train_generator.class_indices.items()}

with open("models/labels.txt", "w") as f:
    for i in range(len(labels)):
        f.write(labels[i] + "\n")

print("Labels saved.")

# -------------------------------
# CNN Model
# -------------------------------
model = Sequential([

    Conv2D(32, (3,3), activation="relu", input_shape=(64,64,3)),
    MaxPooling2D(2,2),

    Conv2D(64,(3,3),activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128,(3,3),activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(256,activation="relu"),
    Dropout(0.5),

    Dense(train_generator.num_classes,activation="softmax")

])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# Save Best Model
# -------------------------------
checkpoint = ModelCheckpoint(
    "models/sign_language_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# -------------------------------
# Train
# -------------------------------
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=3,
    callbacks=[checkpoint]
)

print("\nTraining Completed!")

print("Model Saved -> models/sign_language_model.h5")
print("Labels Saved -> models/labels.txt")