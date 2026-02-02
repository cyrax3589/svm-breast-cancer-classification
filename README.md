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
```

---

## What I Learned

While working with SVM, I learned that the margin is the gap between the decision boundary and the nearest data points from each class. These nearest points are called support vectors, and the main goal of SVM is to maximize this margin so the model can generalize better on unseen data.

I also understood the difference between linear and RBF kernels. A linear kernel works well when the data can be separated using a straight line or flat plane. It is simpler and faster. The RBF kernel, on the other hand, is useful when the data is more complex and not linearly separable, as it allows the model to create non-linear decision boundaries.

The C parameter helped me understand the balance between accuracy and generalization. A smaller C allows the model to tolerate some misclassifications in order to keep a wider margin, while a larger C tries to classify every training point correctly, which can sometimes lead to overfitting.

Gamma plays an important role in how flexible the decision boundary is when using the RBF kernel. A lower gamma value results in smoother boundaries with broader influence from each data point, while a higher gamma focuses more tightly on individual points and can capture complex patterns but may overfit.

Finally, I learned why feature scaling is necessary for SVM. Since SVM depends on distance calculations, features with larger numerical ranges can dominate the model if scaling is not applied. Standardizing the data ensures that all features contribute equally to the decision process.
