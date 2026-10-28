# Oral Disease Detection

A deep learning project that analyzes oral images to detect potential oral health conditions using a trained ResNet model. The project includes a Gradio web interface where users can upload an image and view the prediction results.

## Features

- Upload an oral image for analysis.
- Detect multiple conditions using individual detection thresholds.
- Display prediction scores for each condition.
- Visualize the results through an interactive web interface.
- View sample images in the interface.

## Technologies Used

- Python
- TensorFlow / Keras
- ResNet
- NumPy
- Pillow (PIL)
- Gradio

## Project Structure

Oral-Disease/
├── app/
│   ├── app.py
│   ├── requirements.txt
│   ├── resnet_final.keras
│   └── thresholds.json
├── imgs/
│   └── Sample images for the UI
├── oral-disease.ipynb
└── requirements.txt

## Installation and Usage

### 1. Clone the repository

git clone https://github.com/dareenhdeya/Oral-Disease.git
cd Oral-Disease

### 2. Install dependencies

pip install -r requirements.txt

### 3. Run the application

cd app
python app.py


### 4. Use the application

Open the local URL displayed in the terminal, upload an oral image, and click Analyze to view the prediction results.

## Model

The application uses a trained ResNet model saved as resnet_final.keras. Each condition has its own detection threshold, which is used to determine whether it is detected.

## Notebook

The oral-disease.ipynb notebook contains the project experiments and model development work.
