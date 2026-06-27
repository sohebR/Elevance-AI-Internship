import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import pandas as pd

from predictor import predict


class NationalityGUI:

    def __init__(self, root):

        self.root = root
        self.root.title("Nationality Detection Model")
        self.root.geometry("900x650")
        self.root.resizable(False, False)
        self.root.configure(bg="#f0f0f0")

        self.image_path = None
        self.photo = None

        title = tk.Label(
            root,
            text="Nationality Detection Model",
            font=("Arial", 22, "bold"),
            bg="#f0f0f0",
            fg="#1E3A8A"
        )

        title.pack(pady=15)

        # ---------------- Image Preview ---------------- #

        self.preview = tk.Label(
            root,
            text="No Image Selected",
            width=40,
            height=18,
            relief="solid",
            bg="white"
        )

        self.preview.pack()

        # ---------------- Buttons ---------------- #

        button_frame = tk.Frame(root, bg="#f0f0f0")
        button_frame.pack(pady=15)

        upload_btn = tk.Button(
            button_frame,
            text="Upload Image",
            command=self.upload_image,
            width=18,
            bg="#4CAF50",
            fg="white"
        )

        upload_btn.grid(row=0, column=0, padx=10)

        predict_btn = tk.Button(
            button_frame,
            text="Predict",
            command=self.predict_result,
            width=18,
            bg="#2196F3",
            fg="white"
        )

        predict_btn.grid(row=0, column=1, padx=10)

        save_btn = tk.Button(
            button_frame,
            text="Save Results",
            command=self.save_result,
            width=18,
            bg="#9C27B0",
            fg="white"
        )
        save_btn.grid(row=0, column=2, padx=10)

        clear_btn = tk.Button(
            button_frame,
            text="Clear",
            command=self.clear_all,
            width=18,
            bg="#FF9800",
            fg="white"
        )
        clear_btn.grid(row=0, column=3, padx=10)

        exit_btn = tk.Button(
            button_frame,
            text="Exit",
            command=self.exit_app,
            width=18,
            bg="#F44336",
            fg="white"
        )
        exit_btn.grid(row=0, column=4, padx=10)

        # ---------------- Results ---------------- #

        result_frame = tk.LabelFrame(
            root,
            text="Prediction Results",
            font=("Arial", 12, "bold"),
            padx=15,
            pady=15
        )

        result_frame.pack(fill="both", padx=30, pady=10)

        self.result_text = tk.Text(
            result_frame,
            height=12,
            width=70,
            font=("Consolas", 12)
        )

        self.result_text.pack()

    # ------------------------------------------------ #

    def upload_image(self):

        file_path = filedialog.askopenfilename(

            filetypes=[
                ("Images", "*.jpg *.jpeg *.png")
            ]

        )

        if not file_path:
            return

        self.image_path = file_path

        image = Image.open(file_path)

        image.thumbnail((350, 350))

        self.photo = ImageTk.PhotoImage(image)

        self.preview.config(image=self.photo, text="")

    # ------------------------------------------------ #

    def predict_result(self):

        if self.image_path is None:

            messagebox.showwarning(
                "Warning",
                "Please upload an image."
            )

            return

        result = predict(self.image_path)

        self.result_text.delete("1.0", tk.END)

        output = ""

        for key, value in result.items():

            output += f"{key:<15}: {value}\n"

        self.result_text.insert(tk.END, output)
        # ------------------------------------------------ #

    def save_result(self):

        content = self.result_text.get("1.0", tk.END).strip()

        if content == "":
            messagebox.showwarning(
                "Warning",
                "No prediction results to save."
            )
            return

        data = {}

        for line in content.split("\n"):

            if ":" in line:
                key, value = line.split(":", 1)
                data[key.strip()] = value.strip()

        df = pd.DataFrame([data])

        os.makedirs("outputs", exist_ok=True)

        filename = os.path.join(
            "outputs",
            "prediction_results.csv"
        )

        if os.path.exists(filename):

            old = pd.read_csv(filename)

            df = pd.concat([old, df], ignore_index=True)

        df.to_csv(filename, index=False)

        messagebox.showinfo(
            "Saved",
            f"Results saved to\n{filename}"
        )

    # ------------------------------------------------ #

    def clear_all(self):

        self.image_path = None

        self.preview.config(
            image="",
            text="No Image Selected"
        )

        self.photo = None

        self.result_text.delete("1.0", tk.END)

    # ------------------------------------------------ #

    def exit_app(self):

        self.root.destroy()

if __name__ == "__main__":

    root = tk.Tk()

    app = NationalityGUI(root)

    root.mainloop()