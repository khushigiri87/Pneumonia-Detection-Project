# 🩺 Pneumonia Detection Using Convolutional Neural Network (CNN)

## 📌 Project Overview
This project focuses on detecting **Pneumonia** from chest X-ray images using a deep learning model based on **Convolutional Neural Networks (CNN)**.  
The model classifies X-ray images into two categories:
- **Pneumonia**
- **Normal**

This automated system can assist medical professionals in early and accurate diagnosis.

---

## 🎯 Objective
To build an automated and reliable deep learning system that accurately detects Pneumonia from chest X-ray images and supports early medical diagnosis.

---

## 📂 Dataset
- **Chest X-ray Images (Pneumonia & Normal)**
- **Source:** Kaggle
- Dataset split:
  - Training
  - Validation
  - Testing

---

## 🧠 Model Architecture
- Convolutional Neural Network (CNN)
- Conv2D and MaxPooling layers
- Fully connected Dense layers
- Sigmoid activation function for binary classification

---

## 📄 Project Documents
- **Model Details:** `Project model.pdf`
- **Case Study:** `Case Study.pdf`
- **Book Chapter:** `Pneumonia Detection Using Convolutional Neural Networks on Chest X.pdf`

---

## ✨ Features
- Image preprocessing and normalization
- Data augmentation
- Model training and fine-tuning
- Evaluation on test dataset
- Accuracy and Loss visualization
- Confusion Matrix
- Prediction on newly uploaded X-ray images
- Incremental retraining (demonstration purpose)

---

## 🛠️ Technologies Used
- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab

---

## ▶️ How to Run the Project
1. Open the notebook in **Google Colab**
2. Upload the dataset into the `dataset` folder
3. Run all cells to train the model
4. Trained models will be saved in the `model` folder
5. Results and graphs will be saved in the `results` folder
6. Use the prediction code to test new chest X-ray images

---

## 📊 Results
- **Training Accuracy:** ~93% – 95%
- **Validation Accuracy:** Varies due to small validation dataset
- Performs well on clear Pneumonia cases

---

## 📉 Evaluation Metrics
- Accuracy vs Loss graphs
- Confusion Matrix
- Model performance on unseen test data

---

## ⚠️ Limitations
- Reduced accuracy on early-stage Pneumonia
- Sensitive to low-quality or blurred X-ray images
- Limited performance on pediatric chest X-rays
- Incremental retraining with a single image is for demonstration only

---

## 🚀 Future Scope
- Training with larger and balanced datasets
- Deployment as a web or mobile application
- Multi-disease classification (COVID-19, TB, etc.)
- Explainable AI using Grad-CAM

---

## 👩‍💻 Author
**Khushi Giri**  
3rd Year, 5th Semester  
B.Tech – Computer Science & Engineering  
Galgotias University
