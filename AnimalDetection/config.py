"""
config.py
-----------
Stores application configuration, colors, supported animal classes,
and other constants.
"""

# YOLO model path
MODEL_PATH = "models/yolov8n.pt"

# Output folders
OUTPUT_FOLDER = "output"
LOG_FOLDER = "logs"

# Colors (OpenCV uses BGR)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)

# Font settings
FONT_SCALE = 0.7
THICKNESS = 2

# Animals available in YOLOv8 COCO model
SUPPORTED_ANIMALS = [
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe"
]

# Carnivorous animals
CARNIVORES = [
    "cat",
    "dog",
    "bear",

    "lion",
    "tiger",
    "leopard",
    "cheetah",
    "wolf",
    "fox",
    "hyena",
    "jaguar",
    "lynx",
    "panther",
    "crocodile",
    "alligator",
    "eagle",
    "hawk",
    "owl",
    "snake"
]

# Window settings
WINDOW_TITLE = "Animal Detection System"

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 750

# CSV Log file
CSV_FILE = "logs/detections.csv"