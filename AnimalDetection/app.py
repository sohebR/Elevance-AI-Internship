"""
app.py
Professional Animal Detection System
"""

import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
from PIL import Image, ImageTk

from detector import detect_image, detect_video
from config import WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT


# -----------------------------
# Main Window
# -----------------------------

root = tk.Tk()
root.title(WINDOW_TITLE)
root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
root.configure(bg="#ECEFF1")
root.resizable(False, False)


# -----------------------------
# Variables
# -----------------------------

status_var = tk.StringVar()
status_var.set("Ready")

animal_var = tk.StringVar()
animal_var.set("No detections yet")

carnivore_var = tk.StringVar()
carnivore_var.set("0")

preview_image = None


# -----------------------------
# Header
# -----------------------------

header = tk.Frame(root, bg="#1565C0", height=70)
header.pack(fill="x")

title = tk.Label(
    header,
    text="🐾 Animal Detection System",
    font=("Segoe UI", 22, "bold"),
    fg="white",
    bg="#1565C0"
)

title.pack(pady=15)


# -----------------------------
# Main Layout
# -----------------------------

main_frame = tk.Frame(root, bg="#ECEFF1")
main_frame.pack(fill="both", expand=True, padx=15, pady=15)


# =============================
# LEFT PANEL
# =============================

left_panel = tk.Frame(main_frame, bg="white", relief="ridge", bd=2)
left_panel.pack(side="left", fill="both", expand=True, padx=(0,10))


preview_label = tk.Label(
    left_panel,
    text="Image / Video Preview",
    font=("Segoe UI",16,"bold"),
    bg="white"
)

preview_label.pack(pady=10)


preview_canvas = tk.Label(
    left_panel,
    bg="#DDDDDD",
    width=80,
    height=30
)

preview_canvas.pack(padx=15,pady=10)


# =============================
# RIGHT PANEL
# =============================

right_panel = tk.Frame(main_frame, bg="white", width=300, relief="ridge", bd=2)
right_panel.pack(side="right", fill="y")

right_panel.pack_propagate(False)


summary_title = tk.Label(
    right_panel,
    text="Detection Summary",
    font=("Segoe UI",18,"bold"),
    bg="white"
)

summary_title.pack(pady=20)


animals_heading = tk.Label(
    right_panel,
    text="Animals Detected",
    font=("Segoe UI",12,"bold"),
    bg="white"
)

animals_heading.pack()


animals_box = tk.Text(
    right_panel,
    width=30,
    height=15,
    font=("Consolas",10),
    state="disabled"
)

animals_box.pack(pady=10)


carnivore_title = tk.Label(
    right_panel,
    text="Carnivorous Animals",
    font=("Segoe UI",12,"bold"),
    bg="white"
)

carnivore_title.pack()


carnivore_label = tk.Label(
    right_panel,
    textvariable=carnivore_var,
    font=("Segoe UI",28,"bold"),
    fg="red",
    bg="white"
)

carnivore_label.pack(pady=10)


# =============================
# Buttons
# =============================

button_frame = tk.Frame(right_panel, bg="white")
button_frame.pack(pady=15)


image_button = ttk.Button(
    button_frame,
    text="📷 Detect Image"
)

image_button.pack(fill="x", pady=5)


video_button = ttk.Button(
    button_frame,
    text="🎥 Detect Video"
)

video_button.pack(fill="x", pady=5)


# =============================
# Status Bar
# =============================

status = tk.Label(
    root,
    textvariable=status_var,
    anchor="w",
    bg="#263238",
    fg="white",
    padx=10
)

status.pack(fill="x")

# =====================================================
# Helper Functions
# =====================================================

def update_animals_box(animals):
    """
    Updates the detection summary.
    """

    animals_box.config(state="normal")
    animals_box.delete("1.0", tk.END)

    if len(animals) == 0:
        animals_box.insert(tk.END, "No animals detected")
    else:
        unique_animals = sorted(set(animals))

        for animal in unique_animals:
            animals_box.insert(tk.END, f"• {animal.title()}\n")

    animals_box.config(state="disabled")


def show_preview(image_path):
    """
    Displays detected image inside GUI.
    """

    global preview_image

    image = Image.open(image_path)

    image.thumbnail((700, 520))

    preview_image = ImageTk.PhotoImage(image)

    preview_canvas.configure(image=preview_image)
    preview_canvas.image = preview_image


# =====================================================
# Image Detection
# =====================================================

def image_detection():

    filename = filedialog.askopenfilename(

        title="Select Image",

        filetypes=[

            ("Images", "*.jpg *.jpeg *.png *.bmp")

        ]

    )

    if filename == "":
        return

    status_var.set("Processing Image...")

    root.update()

    try:

        output_path, animals, carnivores = detect_image(filename)

        show_preview(output_path)

        update_animals_box(animals)

        carnivore_var.set(str(carnivores))

        status_var.set("Image Detection Completed")

        messagebox.showinfo(

            "Detection Complete",

            f"Total Animals : {len(animals)}\n\n"
            f"Carnivorous Animals : {carnivores}"

        )

    except Exception as e:

        messagebox.showerror(

            "Error",

            str(e)

        )

        status_var.set("Ready")

# =====================================================
# Video Detection
# =====================================================

def play_video(video_path):
    """
    Plays the detected video using OpenCV.
    Press 'Q' to close the video window.
    """

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow("Detected Video", frame)

        key = cv2.waitKey(25)

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def video_detection():

    filename = filedialog.askopenfilename(

        title="Select Video",

        filetypes=[

            ("Videos", "*.mp4 *.avi *.mov *.mkv")

        ]

    )

    if filename == "":
        return

    status_var.set("Processing Video...")

    root.update()

    try:

        output_path, animals, carnivores = detect_video(filename)

        update_animals_box(animals)

        carnivore_var.set(str(carnivores))

        status_var.set("Opening Processed Video...")

        play_video(output_path)

        status_var.set("Video Detection Completed")

        messagebox.showinfo(

            "Detection Complete",

            f"Animals Detected : {len(animals)}\n\n"
            f"Carnivorous Animals : {carnivores}"

        )

    except Exception as e:

        messagebox.showerror(

            "Error",

            str(e)

        )

        status_var.set("Ready")

# =====================================================
# Thread Functions
# =====================================================

def run_image_detection():
    """
    Runs image detection in a separate thread.
    """
    threading.Thread(
        target=image_detection,
        daemon=True
    ).start()


def run_video_detection():
    """
    Runs video detection in a separate thread.
    """
    threading.Thread(
        target=video_detection,
        daemon=True
    ).start()


# =====================================================
# Connect Buttons
# =====================================================

image_button.config(
    command=run_image_detection
)

video_button.config(
    command=run_video_detection
)


# =====================================================
# Welcome Message
# =====================================================

animals_box.config(state="normal")

animals_box.insert(
    tk.END,
    "Welcome!\n\n"
    "1. Click 'Detect Image' to analyze an image.\n\n"
    "2. Click 'Detect Video' to analyze a video.\n\n"
    "Detected animals will appear here."
)

animals_box.config(state="disabled")


# =====================================================
# Close Application Properly
# =====================================================

def on_closing():

    if messagebox.askokcancel(
        "Exit",
        "Do you want to exit the application?"
    ):
        root.destroy()


root.protocol("WM_DELETE_WINDOW", on_closing)


# =====================================================
# Start GUI
# =====================================================

root.mainloop()