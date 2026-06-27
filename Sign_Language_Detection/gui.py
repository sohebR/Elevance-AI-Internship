import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from datetime import datetime

from image_predict import predict_image
from realtime import start_camera

# -----------------------------
# Time Restriction
# -----------------------------
current = datetime.now().time()

start = datetime.strptime("00:00", "%H:%M").time()
end = datetime.strptime("22:00", "%H:%M").time()

if not (start <= current <= end):
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(
        "Application Closed",
        "This application only works between\n\n6:00 PM and 10:00 PM"
    )
    exit()

# -----------------------------
# Window
# -----------------------------
root = tk.Tk()

root.title("Sign Language Detection")

root.geometry("700x600")

root.configure(bg="#F4F4F4")

# -----------------------------
# Labels
# -----------------------------
title = tk.Label(
    root,
    text="Sign Language Detection",
    font=("Arial", 22, "bold"),
    bg="#F4F4F4"
)

title.pack(pady=20)

image_label = tk.Label(root, bg="#F4F4F4")
image_label.pack()

prediction_label = tk.Label(
    root,
    text="Prediction :",
    font=("Arial", 16),
    bg="#F4F4F4"
)

prediction_label.pack(pady=15)

# -----------------------------
# Upload Image
# -----------------------------
def upload():

    path = filedialog.askopenfilename(
        filetypes=[
            ("Images","*.jpg *.jpeg *.png")
        ]
    )

    if path == "":
        return

    img = Image.open(path)

    img.thumbnail((300,300))

    photo = ImageTk.PhotoImage(img)

    image_label.configure(image=photo)

    image_label.image = photo

    label, confidence = predict_image(path)

    prediction_label.config(
        text=f"Prediction : {label}\nConfidence : {confidence*100:.2f}%"
    )

# -----------------------------
# Start Camera
# -----------------------------
def camera():
    start_camera()

# -----------------------------
# Buttons
# -----------------------------
upload_btn = tk.Button(
    root,
    text="Upload Image",
    font=("Arial",14),
    command=upload,
    width=18,
    bg="#4CAF50",
    fg="white"
)

upload_btn.pack(pady=10)

camera_btn = tk.Button(
    root,
    text="Start Camera",
    font=("Arial",14),
    command=camera,
    width=18,
    bg="#2196F3",
    fg="white"
)

camera_btn.pack(pady=10)

exit_btn = tk.Button(
    root,
    text="Exit",
    font=("Arial",14),
    command=root.destroy,
    width=18,
    bg="#F44336",
    fg="white"
)

exit_btn.pack(pady=25)
