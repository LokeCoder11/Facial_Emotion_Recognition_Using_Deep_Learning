# Facial_Emotion_Recognition_Using_Deep_Learning


---

**DeepFER** is a robust, real-time facial emotion recognition (*FER*) system leveraging cutting-edge deep learning methods. This project implements both **custom Convolutional Neural Networks (CNNs)** and **Transfer Learning** (using ResNet50V2) to classify seven fundamental human emotions from facial images — optimized for real-world usability with **Streamlit & OpenCV** for instant image and webcam inference.

[![ResNet Model Download](https://img.shields.io/badge/Download-ResNet50V2-blue)](https://drive.google.com/file/d/1j2DsAt-m0D75-7ekgs8EQMLS-9EoPhSU/view?usp=sharing)

---

## 🧭 Overview

- **Goal:** Building a scalable, cross-domain FER system for instant, accurate emotion identification using modern AI.
- **Core Technologies:** CNN, Transfer Learning (ResNet50V2), Streamlit, OpenCV.
- **Applications:** Human-computer interaction, mental health, customer service, and beyond.

---

## 🚩 Problem Statement

Traditional approaches to emotion recognition lack resilience across diverse datasets and are sensitive to real-world variations. **DeepFER** meets this challenge by marrying advanced deep learning with deployable, real-time interfaces, outperforming many handcrafted solutions for facial emotion analysis.

---

## 🤗 Motivation & Background

In sectors like *e-commerce, healthcare, and digital agents*, understanding user emotions is fundamental. DeepFER empowers systems to:
- React instantly to user sentiment during live interactions,
- Offer more empathetic and adaptive feedback,
- Simplify large-scale feedback collection and analysis.

---

## 📚 Dataset

- **Classes:** angry, sad, happy, fear, neutral, disgust, surprise.
- **Images:** Diverse, high-quality, with variations in lighting and background—includes both posed and spontaneous expressions.
- **Annotations:** Each image is labeled by emotion class.
- **Augmentation:** Rotation, scaling, flipping, zoom, brightness—improving generalization.
- **Source:** Aggregated from public datasets and crowd-sourced images.

---

## 🧠 Key Concepts

- Deep feature extraction with CNNs
- Transfer Learning via ResNet50V2 (ImageNet weights)
- Real-time inference (webcam/image upload)
- Data normalization & augmentation
- Intuitive user interaction using Streamlit & OpenCV

---

## ✅ Model Artifacts

The repository includes:

| File                                   | Purpose                                               |
| --------------------------------------- | ----------------------------------------------------- |
| `DeepFER_Facial_Emotion_Recognition_Using_Deep_Learning.ipynb` | End-to-end model development notebook                |
| `best_model.keras`                      | Final trained Keras model                             |
| `cnn_model_new_.h5`                     | Custom CNN checkpoint                                 |
| `ResNet50V2_Model.h5`                   | ResNet50V2 transfer learning model                    |
| `haarcascade_frontalface_default.xml`   | Haar Cascade classifier for OpenCV face detection      |
| `app.py`                               | Streamlit-based single image FER app                  |
| `app2.py`                              | Streamlit + OpenCV webcam real-time FER app           |

[Download ResNet50V2 Model](https://drive.google.com/file/d/1j2DsAt-m0D75-7ekgs8EQMLS-9EoPhSU/view?usp=sharing)

---

## 📊 Evaluation

- **Metrics:** Accuracy, Precision, Recall, F1-Score
- **Visualization:** Confusion matrix, ROC curve, class-wise analysis (implemented with matplotlib)

---

## 🧩 Project Pipeline

1. **Data Exploration**
   - Analyze and visualize class distributions and image samples.

2. **Data Preprocessing**
   - Convert to grayscale, resize (48×48), normalize pixel values.
   - Augment with rotation, flip, zoom, brightness, etc.

3. **Model Construction**
   - **Custom CNN**: Deep architecture with regularization layers.
   - **ResNet50V2**: Transfer learning, with fine-tuning of higher layers.

4. **Training & Optimization**
   - Adam optimizer, categorical crossentropy loss.
   - Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau for best convergence.

5. **Model Evaluation**
   - Detailed reporting: accuracy, F1, confusion matrix, ROC Curve.

6. **Deployment**
   - Streamlit applications for real-time (webcam + image) inference.

---

## 🛠️ Technical Stack

| Category           | Dependency             |
| ------------------ | --------------------- |
| Deep Learning      | TensorFlow, Keras     |
| Image Processing   | OpenCV, Pillow (PIL)  |
| Web Framework      | Streamlit             |
| Visualization      | Matplotlib, Seaborn   |
| Face Detection     | Haar Cascades         |
| Transfer Learning  | ResNet50V2            |

---

## 🚀 Deployment

### 💻 Streamlit Apps

- `app.py`: Upload an image & receive emotion predictions (styled interface).
- `app2.py`: Real-time webcam emotion detection (bounding boxes for faces).

*Launch with:*
streamlit run app.py # For image-based prediction
streamlit run app2.py # For webcam-based real-time FER


---

## 🤖 Supported Emotion Classes

- Angry 😠  
- Disgust 🤢  
- Fear 😨  
- Happy 😊  
- Neutral 😐  
- Sad 😢  
- Surprise 😲  

---

## 💡 Key Features

- **Offline, Local Inference** – No cloud dependency at test time
- **Fast Real-Time Processing** – Suitable for live video streams
- **Lightweight Models** – Fast, low-latency deployment
- **Flexible UI** – Streamlit interface for both images and webcam feeds
- **GPU Support** – Rapid training, easy Google Colab compatibility

---

## 🔥 Learnings & Takeaways

- Advanced CNN and transfer learning techniques
- Data-centric pipeline (augmentation, normalization)
- Real-time model deployment & interactive visualization
- End-to-end user interface creation for deep learning models
- Practical insights for deploying empathetic machine intelligence for real-world impact

---

## 🙋 Author
Lokesh Todi
