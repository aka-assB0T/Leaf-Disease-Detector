import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

class_mapping = {
    0: "Healthy",         # Class 0 represents healthy leaves
    1: "Blight",          # Class 1 represents blight disease
    2: "Rust",            # Class 2 represents rust disease
    3: "Powdery Mildew",  # Class 3 represents powdery mildew
    4: "Scab",            # Class 4 represents scab disease
    5: "Leaf Spot",       # Class 5 represents leaf spot disease
    6: "Early Blight",    # Class 6 represents early blight
    7: "Late Blight",     # Class 7 represents late blight
    8: "Leaf Curl",       # Class 8 represents leaf curl disease
    # Add more classes if needed based on your dataset
}

# Function to predict a new image
def predict_image(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the class
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Map the predicted class to its label
    predicted_label = class_mapping.get(predicted_class, "Unknown")

    return predicted_label, img


# Function to visualize the predicted image
def visualize_prediction(img, predicted_class):
    plt.figure(figsize=(5, 5))  # Set figure size
    plt.imshow(img)  # Display the image
    plt.title(f"Predicted Class: {predicted_class}")  # Show predicted label
    plt.axis('off')  # Hide axes
    plt.show()  # Display the plot

# Load your model
model = load_model("leaf_disease_model.keras")  # Adjust path as needed


# Example of predicting a new image
img_path = "C:/f/PlantVillage/test.jpg"  # Replace with the path to your image
print(f"Using image at: {img_path}")  # Debugging line
predicted_class, img = predict_image(img_path, model)
print(f"Predicted Class: {predicted_class}")

# Visualize the predicted image
visualize_prediction(img, predicted_class)
