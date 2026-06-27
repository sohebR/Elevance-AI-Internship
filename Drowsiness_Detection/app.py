import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os

from detector import detect_image
from video_detector import detect_video

selected_file = None


# -------------------------
# Preview Functions
# -------------------------

def preview_image(path):
    image = Image.open(path)
    image.thumbnail((650, 420))

    photo = ImageTk.PhotoImage(image)

    preview_label.config(image=photo)
    preview_label.image = photo


def preview_video(path):

    cap = cv2.VideoCapture(path)

    success, frame = cap.read()

    cap.release()

    if success:

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(frame)

        image.thumbnail((650, 420))

        photo = ImageTk.PhotoImage(image)

        preview_label.config(image=photo)
        preview_label.image = photo


# -------------------------
# Select Image
# -------------------------

def select_image():

    global selected_file

    file = filedialog.askopenfilename(

        filetypes=[
            ("Images", "*.jpg *.jpeg *.png")
        ]
    )

    if file:

        selected_file = file

        preview_image(file)

        status_label.config(
            text="Image Selected"
        )


# -------------------------
# Select Video
# -------------------------

def select_video():

    global selected_file

    file = filedialog.askopenfilename(

        filetypes=[
            ("Videos", "*.mp4 *.avi *.mov")
        ]
    )

    if file:

        selected_file = file

        preview_video(file)

        status_label.config(
            text="Video Selected"
        )


# -------------------------
# Detection
# -------------------------

def start_detection():

    global selected_file

    if selected_file is None:

        messagebox.showwarning(
            "Warning",
            "Please select an image or video."
        )

        return

    extension = os.path.splitext(selected_file)[1].lower()

    if extension in [".jpg", ".jpeg", ".png"]:

        result = detect_image(selected_file)

    else:

        result = detect_video(selected_file)

    messagebox.showinfo(

        "Detection Result",

        f"""Total Persons : {result['total']}

Sleeping : {result['sleeping']}

Awake : {result['awake']}

Sleeping Ages

{result['ages']}
"""
    )


# -------------------------
# GUI
# -------------------------

root = tk.Tk()

root.title("Drowsiness Detection System")

root.geometry("900x700")

root.resizable(False, False)


title = tk.Label(

    root,

    text="Drowsiness Detection System",

    font=("Arial", 22, "bold")

)

title.pack(pady=15)


button_frame = tk.Frame(root)

button_frame.pack()


image_btn = tk.Button(

    button_frame,

    text="Select Image",

    width=18,

    font=("Arial", 12),

    command=select_image

)

image_btn.grid(row=0, column=0, padx=10)


video_btn = tk.Button(

    button_frame,

    text="Select Video",

    width=18,

    font=("Arial", 12),

    command=select_video

)

video_btn.grid(row=0, column=1, padx=10)


preview_label = tk.Label(

    root,

    width=650,

    height=420,

    bg="gray",

    relief="solid"

)

preview_label.pack(pady=20)


detect_btn = tk.Button(

    root,

    text="Start Detection",

    width=25,

    bg="green",

    fg="white",

    font=("Arial", 13, "bold"),

    command=start_detection

)

detect_btn.pack(pady=10)


status_label = tk.Label(

    root,

    text="No file selected",

    font=("Arial", 12)

)

status_label.pack()


exit_btn = tk.Button(

    root,

    text="Exit",

    width=20,

    font=("Arial", 12),

    command=root.destroy

)

exit_btn.pack(pady=15)


root.mainloop()