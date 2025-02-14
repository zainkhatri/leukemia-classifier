# Leukemia Classifier: AI-Based Early Detection System

## Overview
This project aims to develop an **AI-powered system for early detection and subtype classification of Acute Lymphoblastic Leukemia (ALL)**. Using **3,242 peripheral blood smear images**, our deep learning model will classify blood cells into four categories:  

✅ **Benign**  
✅ **Early Pre-B**  
✅ **Pre-B**  
✅ **Pro-B ALL**  

We are integrating **uncertainty quantification** and **explainability (Grad-CAM)** to enhance trust and clinical usability.

---

## The Next Steps

### 1️⃣ **Data Preparation**  
- Load **Blood Cells Cancer (ALL) dataset** (sourced from Taleqani Hospital, Iran).  
- Apply **normalization, augmentation (rotation, scaling, color jittering)**.  
- Ensure **patient-wise separation** to avoid data leakage.  

### 2️⃣ **Model Development**  
- Implement **ResNet50-based CNN** for classification.  
- Add **Monte Carlo Dropout** for uncertainty estimation.  
- Train an **ensemble of 5 models** to improve reliability.  

### 3️⃣ **Interpretability & Evaluation**  
- Apply **Grad-CAM** for visual heatmaps.  
- Validate uncertainty estimates using **Expected Calibration Error (ECE)**.  
- Measure accuracy, F1-score, and **ROC-AUC curves** for performance analysis.  

### 4️⃣ **Optimization & Testing**  
- Ensure **<2 seconds per image inference speed**.  
- Run **extensive testing** on independent validation sets.  

---

## Status  
🚀 **In Progress – Model Implementation Phase**  