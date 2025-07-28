# 🧠 Visual Recognition of Hair Disorders: Machine Learning for Classification
# 📌 Project Overview
This project employs deep learning techniques using Convolutional Neural Networks (CNNs) to classify various hair diseases from images. The objective is to support dermatological diagnosis through automated, accurate, and fast identification of conditions such as alopecia, psoriasis, folliculitis, and others using clinical images.

We implemented and compared three CNN architectures:

MobileNetV2

InceptionV3

DenseNet169

# 🎯 Aim of the Project
To develop an efficient machine learning-based image classification system capable of diagnosing 10 distinct hair diseases using dermatological images. The goal is to augment clinical decision-making, reduce diagnosis time, and provide a non-invasive screening tool.

# 📊 Dataset
Source: Kaggle

Total Images: 12,000 (colored dermatological images)

# Classes:

Alopecia Areata

Contact Dermatitis

Folliculitis

Head Lice

Lichen Planus

Male Pattern Baldness

Psoriasis

Seborrheic Dermatitis

Telogen Effluvium

Tinea Capitis

# 📂 Dataset Split
Dataset Type	Images per Class	Total Images
Training	960	9,600
Validation	120	1,200
Testing	120	1,200

# 🧼 Preprocessing:
Resizing: All images resized to 224x224 pixels

Normalization: Scaled pixel values to [0,1]

Augmentation: Rotation, width/height shift, zoom, flip to increase generalizability

# 🧪 Models & Methods
✅ MobileNetV2
Lightweight CNN optimized for mobile/edge use

Used pre-trained ImageNet weights (Transfer Learning)

Added custom layers: Dense (512, ReLU) + Dropout + Softmax

Test Accuracy: ~98%

✅ InceptionV3
Designed for multi-scale image analysis

Pre-trained weights used, layers frozen

Layers added: Flatten → Dense(512) + Dropout → Softmax

Test Accuracy: ~90.24%

✅ DenseNet169
Dense connectivity, reuse of feature maps

Pre-trained ImageNet weights + custom head

Layers: GlobalAvgPooling → Dense(512, ReLU) + Dropout → Softmax

Test Accuracy: 99.67%

# ⚙️ Functions and Why They Were Used
Functionality	Purpose
Conv2D, MaxPooling2D	Extract spatial features
GlobalAveragePooling2D	Reduce feature map size before Dense
Dense with ReLU	Learn complex non-linear patterns
Dropout	Prevent overfitting
Softmax	Output probability for multi-class classification
Adam Optimizer	Efficient learning with adaptive learning rates
Categorical Crossentropy	Suitable for multi-class classification
ImageDataGenerator	Augment images and balance dataset

# 📈 Results & Insights
Model	Test Accuracy	F1 Score	Notes
MobileNetV2	~98%	High	Good speed and performance trade-off
InceptionV3	~90.24%	Good	Slightly lower accuracy, good precision
DenseNet169	~99.67%	High	Best overall model, high accuracy and generalization

Confusion matrices show strong class-level prediction accuracy

Precision-Recall Curves confirm the model's reliability, especially in medical use-cases

Training curves (loss vs. accuracy) confirm proper convergence

# 🚀 Deployment (Optional Step Completed)
The best model (DenseNet169) can be deployed using Flask for web-based usage:

HTML form for image upload

Backend: Flask API loads model and predicts class

Frontend: Displays predicted disease class

# 🔮 Future Scope
Expand Dataset: Include more hair/scalp conditions and diverse populations

Model Explainability: Use Grad-CAM or LIME for interpretability

Integration with EHRs: Combine image diagnosis with patient metadata

Mobile App: Real-time diagnosis using smartphone cameras

3D Imaging Support: Include dermoscopy and trichoscopy data

# 👩‍💻 Tech Stack
Component	Tool
Language	Python
Libraries	TensorFlow, Keras, NumPy, Matplotlib, Scikit-learn
Deployment	Flask
Visualization	Confusion matrix, PR curves, Accuracy/Loss plots

# 📌 How to Run
Clone the repository

Place dataset in the required directory

Train models using provided Jupyter notebooks

To deploy:

bash
Copy
Edit
python app.py
Open browser at http://127.0.0.1:8080 and test predictions

# 👥 Contributors
Gousia Parvin Patthan

Rithika Mahareddy

Harika Kakumanu

