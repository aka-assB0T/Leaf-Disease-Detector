import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance
import cv2
import os
import random
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class LeafDiseaseApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # Window settings
        self.title("Leaf Disease Detection")
        self.geometry("800x400+300+200")
        self.configure(bg="#f0f8ff")  # Light blue background

        # Create layout frames for the image and buttons
        self.create_layout()

        # Placeholder for the loaded image and OpenCV image
        self.loaded_image = None
        self.cv_image = None
        self.original_image = None

        # Set the dataset directory for random image selection
        self.dataset_dir = "C:/f/PlantVillage/PlantVillage"  # Set your dataset directory here

        # Load the custom-trained model for leaf disease detection
        self.model = load_model('leaf_disease_detection_model.keras')

    def create_layout(self):
        """Create the layout with image display and button options."""
        # Left Frame for the image display
        self.left_frame = tk.Frame(self, bg="#e6ffe6", bd=5, relief=tk.RAISED)
        self.left_frame.pack(side="left", fill="both", expand=True)

        # Right Frame for the buttons
        self.right_frame = tk.Frame(self, bg="#ccffcc", bd=5, relief=tk.RAISED)
        self.right_frame.pack(side="right", fill="y", expand=False)

        # Placeholder label for displaying the imported image
        self.image_label = tk.Label(self.left_frame, text="No Image Loaded", bg="#e6ffe6", font=("Helvetica", 14))
        self.image_label.pack(padx=10, pady=10, expand=True)

        # Create buttons for the functionalities in the right frame
        self.create_buttons()

    def create_buttons(self):
        """Create the UI buttons and layout."""
        button_style = {
            'bg': '#4CAF50',  # Green background
            'fg': 'white',  # White text
            'font': ('Helvetica', 12, 'bold'),
            'padx': 10,
            'pady': 5,
            'activebackground': '#45a049',  # Darker green on hover
            'bd': 2,
            'relief': tk.RAISED
        }

        # Button creation
        buttons = [
            ("Import Image", self.import_image),
            ("Load Random Image", self.load_random_image),
            ("Image Enhancement", self.enhance_image),
            ("Disease Prediction", self.predict_disease),
            ("Reset", self.reset),
            ("Exit", self.exit_app)
        ]

        for text, command in buttons:
            btn = tk.Button(self.right_frame, text=text, command=command, **button_style)
            btn.pack(pady=10, fill="x")

    def import_image(self):
        """Handle importing an image from the local file system."""
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if file_path:
            self.loaded_image = Image.open(file_path)
            self.cv_image = cv2.imread(file_path)
            self.original_image = self.loaded_image.copy()
            self.display_image(self.loaded_image)
            messagebox.showinfo("Success", "Image imported successfully!")
        else:
            messagebox.showwarning("Warning", "No image selected!")

    def load_random_image(self):
        """Randomly select and load an image from the dataset directory."""
        if os.path.exists(self.dataset_dir):
            image_files = [os.path.join(root, file)
                           for root, dirs, files in os.walk(self.dataset_dir)
                           for file in files if file.endswith((".jpg", ".png", ".jpeg"))]

            if image_files:
                random_image_path = random.choice(image_files)
                self.loaded_image = Image.open(random_image_path)
                self.cv_image = cv2.imread(random_image_path)
                self.original_image = self.loaded_image.copy()
                self.display_image(self.loaded_image)
                messagebox.showinfo("Success", f"Random image loaded from: {random_image_path}")
            else:
                messagebox.showwarning("Warning", "No image files found in the dataset directory!")
        else:
            messagebox.showwarning("Warning", f"Dataset directory not found: {self.dataset_dir}")

    def display_image(self, img):
        """Display the imported image in the main window."""
        img = img.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.config(image=img_tk, text="")
        self.image_label.image = img_tk

    def enhance_image(self):
        """Apply basic image enhancement techniques."""
        if self.loaded_image and self.cv_image is not None:
            self.enhancement_window = tk.Toplevel(self)
            self.enhancement_window.title("Enhancement Options")
            self.enhancement_window.configure(bg="#ccffcc")

            # Set a fixed size and position, e.g., 300x300 pixels, positioned 850 pixels right and 100 pixels down
            self.enhancement_window.geometry("270x275+900+240")

            button_style = {
                'bg': '#4CAF50',  # Green background
                'fg': 'white',  # White text
                'font': ('Helvetica', 12, 'bold'),
                'padx': 10,
                'pady': 5,
                'activebackground': '#45a049',  # Darker green on hover
                'bd': 2,
                'relief': tk.RAISED
            }

            buttons = [
                ("Enhance Brightness", self.enhance_brightness),
                ("Enhance Contrast", self.enhance_contrast),
                ("Sharpen Image", self.sharpen_image),
                ("Apply Histogram Equalization", self.histogram_equalization),
                ("Show Original Image", self.show_original_image)
            ]

            for text, command in buttons:
                btn = tk.Button(self.enhancement_window, text=text, command=command, **button_style)
                btn.pack(pady=5)

        else:
            messagebox.showwarning("Warning", "No image to enhance! Please import or load an image first.")

    def show_original_image(self):
        """Display the original image."""
        if self.original_image:
            self.display_image(self.original_image)
        else:
            messagebox.showwarning("Warning", "No original image to show!")

    def enhance_brightness(self):
        """Increase the brightness of the image using Pillow."""
        enhancer = ImageEnhance.Brightness(self.loaded_image)
        bright_img = enhancer.enhance(1.5)
        self.display_image(bright_img)

    def enhance_contrast(self):
        """Enhance contrast of the image using Pillow."""
        enhancer = ImageEnhance.Contrast(self.loaded_image)
        contrast_img = enhancer.enhance(1.5)
        self.display_image(contrast_img)

    def sharpen_image(self):
        """Sharpen the image using Pillow."""
        enhancer = ImageEnhance.Sharpness(self.loaded_image)
        sharp_img = enhancer.enhance(2.0)
        self.display_image(sharp_img)

    def histogram_equalization(self):
        """Enhance the image contrast using OpenCV's histogram equalization."""
        gray_img = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        equalized_img = cv2.equalizeHist(gray_img)
        equalized_img = cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2RGB)
        img_pil = Image.fromarray(equalized_img)
        self.display_image(img_pil)

    def predict_disease(self):
        """Use the custom-trained model to predict the disease."""
        if self.loaded_image:
            processed_img = self.preprocess_image_for_model(self.loaded_image)
            preds = self.model.predict(processed_img)
            predicted_class = np.argmax(preds, axis=1)

            class_mapping = {
                0: "Healthy",
                1: "Blight",
                2: "Rust",
                3: "Powdery Mildew",
                4: "Scab",
                5: "Leaf Spot",
                6: "Early Blight",
                7: "Late Blight",
                8: "Leaf Curl"
            }
            disease_name = class_mapping.get(predicted_class[0], "Unknown Disease")
            result = f"Disease: {disease_name} with confidence {preds[0][predicted_class[0]] * 100:.2f}%"
            messagebox.showinfo("Disease Prediction", result)
        else:
            messagebox.showwarning("Warning", "No image to predict! Please import an image first.")

    def preprocess_image_for_model(self, img):
        """Preprocess the image to make it compatible with the custom model."""
        img_resized = img.resize((300, 300))  # Adjusted to 300x300 to match model input size
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array

    def reset(self):
        """Reset the application state."""
        self.loaded_image = None
        self.original_image = None
        self.image_label.config(image="", text="No Image Loaded")
        messagebox.showinfo("Reset", "Application reset successfully!")

    def exit_app(self):
        """Exit the application."""
        self.quit()


if __name__ == "__main__":
    app = LeafDiseaseApp()
    app.mainloop()
