# Tomato-Disease-Prediction

🍅 Tomato Disease Prediction System
📌 Overview

The Tomato Disease Prediction System is a deep learning–based web application that detects diseases in tomato leaves using image classification.
The system allows users (farmers, researchers, or students) to upload an image of a tomato leaf and receive:

✅ Disease name

✅ Confidence of prediction

✅ Recommended treatment solutions

✅ Downloadable PDF report

✅ Multilingual support (English / Hindi / Tamil)

This project demonstrates the practical use of Computer Vision and Machine Learning in Agriculture to help with early disease detection and crop management.

🎯 Objectives

Detect tomato leaf diseases automatically using a trained CNN model.

Provide instant diagnosis through a user-friendly web interface.

Assist farmers with treatment recommendations.

Reduce manual inspection effort and improve crop yield.

Showcase an end-to-end ML pipeline from training to deployment.

🛠️ Technologies Used
Category	Tools / Libraries
Programming Language	Python
Deep Learning	TensorFlow / Keras
Frontend Interface	Streamlit
Image Processing	PIL, OpenCV
Data Handling	NumPy, Pandas
Model Visualization	Matplotlib
Report Generation	FPDF
Development Environment	Google Colab, VS Code
📂 Dataset

Dataset contains ~10,000 images of tomato leaves.

Covers 10 classes including healthy and multiple disease categories.

Images are organized into train / validation directories.

Data includes variations in lighting, angle, and background to improve generalization.

🧠 Model Architecture

The model is built using a Convolutional Neural Network (CNN) for image classification.

Key Layers:

Convolutional Layers → Extract visual features

ReLU Activation → Introduce non-linearity

MaxPooling → Reduce dimensionality

Fully Connected Dense Layers → Classification

Softmax Layer → Multi-class prediction

Why CNN?

CNNs are highly effective in detecting:

Texture patterns

Spots and discoloration

Leaf structure anomalies

⚙️ System Workflow

1️⃣ User uploads tomato leaf image
2️⃣ Image is resized and normalized
3️⃣ Preprocessing applied (tensor conversion)
4️⃣ Trained CNN model performs prediction
5️⃣ Disease class identified
6️⃣ Treatment recommendation displayed
7️⃣ PDF report generated (optional download)

💻 Application Features

✔ Upload Image for Instant Prediction
✔ Displays Disease Name & Accuracy
✔ Provides Suggested Treatment
✔ Supports Multiple Languages
✔ Generates Downloadable Diagnosis Report
✔ Clean and Simple Streamlit Interface

📊 Model Performance

Achieved high classification accuracy during validation.

Handles real-world variations effectively.

Generalizes well across unseen leaf images.
