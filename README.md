# SVM – Breast Cancer Classification

## Overview
This project implements a machine learning pipeline for breast cancer classification using Support Vector Machines (SVM). The goal is to classify tumors as benign or malignant using the Breast Cancer Wisconsin dataset provided by Scikit-learn.

The project demonstrates kernel-based classification, feature scaling, hyperparameter tuning, and model evaluation using industry-standard practices.

---

## Dataset
- **Primary Dataset:** Breast Cancer Wisconsin Dataset  
- **Source:** `sklearn.datasets.load_breast_cancer()`

The dataset contains 30 numerical features computed from digitized images of breast mass biopsies.

---

## Objectives
- Load and inspect the dataset
- Normalize feature values using StandardScaler
- Split data into training and testing sets
- Train an SVM model with a linear kernel
- Train an SVM model with an RBF kernel
- Tune hyperparameters using GridSearchCV
- Evaluate the best model using multiple metrics
- Plot ROC curve and compute AUC score
- Save the trained model pipeline for reuse

---

## Tools and Libraries
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Joblib

---

## Methodology
1. Data loading and inspection
2. Feature scaling using StandardScaler
3. Train-test split with stratification
4. Baseline SVM training using linear kernel
5. Non-linear SVM training using RBF kernel
6. Hyperparameter tuning for C and gamma
7. Model evaluation using:
   - Accuracy
   - Confusion Matrix
   - Classification Report
8. ROC curve plotting and AUC calculation
9. Model persistence using Joblib

---

## Results
- The tuned RBF kernel SVM achieved the best classification performance.
- ROC–AUC analysis confirms strong discriminative capability of the model.

### ROC Curve
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/d63a6f72-da72-4cb4-95f6-eacb733018cd" />


---

## Saved Model
The final tuned model (including scaler and SVM) is saved as:

``` bash
svm_breast_cancer_model.pkl
