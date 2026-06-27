import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

from detector import detect_objects


class CarColorGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Car Colour Detection System")
        self.root.geometry("900x700")
        self.root.resizable(False, False)

        self.image_path = None

        # Title
        tk.Label(
            root,
            text="Car Colour Detection System",
            font=("Arial", 22, "bold")
        ).pack(pady=10)

        # Image Preview
        self.image_label = tk.Label(
            root,
            text="No Image Selected",
            width=70,
            height=20,
            relief="solid"
        )
        self.image_label.pack(pady=10)

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        tk.Button(
            btn_frame,
            text="Upload Image",
            width=18,
            command=self.upload_image
        ).grid(row=0, column=0, padx=10)

        tk.Button(
            btn_frame,
            text="Detect",
            width=18,
            command=self.detect
        ).grid(row=0, column=1, padx=10)

        tk.Button(
            btn_frame,
            text="Exit",
            width=18,
            command=root.destroy
        ).grid(row=0, column=2, padx=10)

        # Results
        self.result_label = tk.Label(
            root,
            text="Cars: 0\nPeople: 0\nBlue Cars: 0\nOther Cars: 0",
            font=("Arial", 14)
        )
        self.result_label.pack(pady=15)

    def upload_image(self):

        path = filedialog.askopenfilename(
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png")
            ]
        )

        if not path:
            return

        self.image_path = path

        img = Image.open(path)
        img.thumbnail((700, 450))

        photo = ImageTk.PhotoImage(img)

        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo

    def detect(self):

        if self.image_path is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        output_path, cars, people, blue, other = detect_objects(self.image_path)

        img = Image.open(output_path)
        img.thumbnail((700, 450))

        photo = ImageTk.PhotoImage(img)

        self.image_label.configure(image=photo)
        self.image_label.image = photo

        self.result_label.config(
            text=f"""Cars Detected : {cars}

People Detected : {people}

Blue Cars : {blue}

Other Cars : {other}
"""
        )

        messagebox.showinfo(
            "Success",
            "Detection completed!\nResult saved in outputs/result.jpg"
        )