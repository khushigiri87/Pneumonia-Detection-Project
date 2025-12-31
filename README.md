# 🩺 Pneumonia Detection Using Convolutional Neural Network (CNN)

![Python](https://img.shields.io/badge/Python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Keras](https://img.shields.io/badge/Keras-CNN-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📌 Project Overview
This project focuses on detecting **Pneumonia** from chest X-ray images using a deep learning model based on **Convolutional Neural Networks (CNN)**.  
The system classifies chest X-ray images into two categories:

- 🫁 **Pneumonia**
- ✅ **Normal**

This automated solution helps in **early diagnosis** and supports medical professionals by reducing manual effort.

---

## 🎯 Objective
To design and implement a **robust CNN-based deep learning model** that accurately detects Pneumonia from chest X-ray images.

---

## 📂 Dataset
- **Chest X-ray Images (Pneumonia & Normal)**
- **Source:** Kaggle
- Dataset is divided into:
  - Training set
  - Validation set
  - Test set

> ⚠️ Due to GitHub size limitations, the dataset is **not uploaded** to this repository.

---

## 🧠 Model Architecture
- Convolutional Neural Network (CNN)
- Multiple Conv2D + MaxPooling layers
- Fully connected Dense layers
- Sigmoid activation for binary classification

---

## 📄 Project Documents
All detailed documents are available in the `docs/` folder:

- 📘 **Model Explanation:** `Project model.pdf`
- 📗 **Case Study:** `Case Study.pdf`
- 📙 **Book Chapter:** `Pneumonia Detection Using Convolutional Neural Networks on Chest X.pdf`

---

## ✨ Features
- Image preprocessing & normalization
- Data augmentation for better generalization
- CNN model training & fine-tuning
- Performance evaluation on test data
- Accuracy & loss visualization
- Confusion matrix analysis
- Pneumonia detection on new X-ray images
- Incremental retraining (demo purpose)

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
2. Upload the dataset inside the `dataset/` folder
3. Run all cells sequentially
4. Trained model will be saved in the `model/` folder
5. Graphs and results will be stored in the `results/` folder
6. Use prediction cells to test new chest X-ray images

---

## 📊 Training & Evaluation Results

### 🔹 Training Accuracy
![Training Accuracy](results/accuracy_plot.png)

### 🔹 Training Loss
![Training Loss](results/loss_plot.png)

### 🔹 Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)

---

## 📈 Performance Summary
- **Training Accuracy:** ~93% – 95%
- **Validation Accuracy:** Varies due to limited validation data
- Best performance on clearly visible Pneumonia X-rays

---

## ⚠️ Limitations
- Lower accuracy on early-stage Pneumonia
- Sensitive to blurred or low-resolution X-rays
- Limited performance on pediatric chest X-rays
- Incremental retraining is for demonstration only

---

## 🚀 Future Scope
- Train on larger & balanced datasets
- Deploy as a **web or mobile healthcare app**
- Multi-disease classification (COVID-19, TB, etc.)
- Explainable AI using **Grad-CAM**

---

## 📂 Project Structure
```text
pneumonia-detection-cnn/
│
├── notebooks/
│   └── pneumonia_detection.ipynb
│
├── dataset/
│   └── (not uploaded – Kaggle dataset)
│
├── model/
│   └── cnn_model.h5
│
├── results/
│   ├── accuracy_plot.png
│   ├── loss_plot.png
│   └── confusion_matrix.png
│
├── docs/
│   ├── Project model.pdf
│   ├── Case Study.pdf
│   └── Pneumonia Detection Using Convolutional Neural Networks on Chest X.pdf
│
└── README.md

---

##
 👩‍💻 Author
**Khushi Giri**  
3rd Year, 5th Semester  
B.Tech – Computer Science & Engineering  
Galgotias University
