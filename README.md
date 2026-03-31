# Leaf Disease Detector

This project demonstrates a Leaf Disease Detection System using a Convolutional Neural Network (CNN). 
It is built with TensorFlow/Keras, OpenCV, and Tkinter to detect plant leaf diseases from images.

## Project Description:

This Leaf Disease Detector system utilizes a Convolutional Neural Network (CNN) to predict plant leaf diseases. It was built using TensorFlow/Keras, OpenCV, and Tkinter. The system allows users to upload leaf images, enhance them, and predict whether the leaf is infected with diseases like Blight, Rust, Powdery Mildew, etc. The model is trained using the PlantVillage dataset from Kaggle, which includes images of various leaf diseases.

## Motivation

The goal of this project was to develop a simple machine learning application that can identify plant diseases from leaf images. 
It combines deep learning with a GUI to make the system interactive and easy to use.

## Table of Contents

-   [Features](#features)
-   [Technologies Used](#technologies-used)
-   [Prerequisites](#prerequisites)
-   [Installation & Setup](#installation--setup)
-   [Usage](#usage)
-   [Project Structure](#project-structure)
-   [Configuration](#configuration)
-   [Contributing](#contributing)
-   [Screenshots](#screenshots)
-   [License](#license)
-   [Acknowledgments](#acknowledgments)

## Features

- Leaf Disease Prediction using a trained CNN model
- Image enhancement (brightness, contrast, sharpening)
- GUI-based interaction using Tkinter
- Random image loading from dataset
- Model training and testing scripts included

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- Pillow (PIL)
- Tkinter

## Prerequisites

- Python 3.x
- TensorFlow
- OpenCV
- Pillow

Install dependencies:

pip install tensorflow opencv-python pillow

## Installation & Setup

Follow these steps to get the project up and running on your local machine:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/aka-assB0T/leaf_disease_detector.git
    cd leaf_disease_detector
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment**:
    *   **On Windows**:
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the application, simply execute the `main.py` file:

```bash
python main.py
```


Then:
- Import or load an image
- Apply enhancements if needed
- Click "Disease Prediction" to get results

## Project Structure

```
.
├── .gitignore               # Specifies intentionally untracked files to ignore
├── main.py                  # Main application entry point, handles UI setup and navigation
├── preprocessing.py         # To process image files and folders categorically
├── train_model.py           # To build a keras model
├── test_model.py            # To test a model performance and working citeria (optional)
├── README.md                # This README file
└── requirements.txt         # Lists Python dependencies
```

## Important Notes

- The trained model file (.keras) is not included due to size limitations [Update: Uploaded].
- You need to place the trained model in the project root directory.
- MUST Update dataset path in the code if necessary.

## Future Improvements

- Improve model accuracy
- Add more disease classes
- Deploy as web application
- Add real-time camera detection

## **Screenshots**
Why not run and see yourself?


## License
This project is licensed under the MIT License - see the [LICENSE](./assets/LICENSE) file for details.


## Acknowledgments

*   This project was developed as a university (North Western University, Khulna) lab project by [aka-assB0T](https://github.com/aka-assB0T).
