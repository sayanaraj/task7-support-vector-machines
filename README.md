# task7-support-vector-machines

Objective

This task demonstrates the application of Support Vector Machines (SVM) for binary classification using the Breast Cancer Wisconsin dataset. It includes training SVM classifiers with both linear and RBF kernels, visualizing decision boundaries, tuning hyperparameters (C and gamma), and evaluating model performance through cross-validation.

Dataset

Name: Breast Cancer Wisconsin (Diagnostic) Dataset  
Source: Kaggle  
File: breast-cancer.csv  
Target Column: diagnosis (B for benign, M for malignant)

Features Used

All numeric features from the dataset were used after dropping irrelevant columns. The 'id' and unnamed columns were removed. The 'diagnosis' column was encoded as binary (B = 0, M = 1).

Libraries Used

pandas  
numpy  
matplotlib  
seaborn  
scikit-learn

Tasks Performed

1. Data Preparation  
The dataset was loaded and cleaned. The target variable was encoded, and feature values were standardized using StandardScaler. The data was split into 80 percent training and 20 percent testing sets.

2. Model Training  
Two SVM models were trained using scikit-learn's SVC class: one with a linear kernel and another with an RBF (Gaussian) kernel.

3. Model Evaluation  
Both models were evaluated using accuracy, confusion matrix, precision, recall, F1-score, and ROC-AUC metrics. ROC curves and confusion matrices were visualized.

4. Hyperparameter Tuning  
GridSearchCV was used to perform hyperparameter tuning for C and gamma on the RBF kernel. A five-fold cross-validation was performed during the search.

5. Cross-Validation  
The best model from the grid search was further evaluated using five-fold cross-validation. The average accuracy across folds was reported.

6. Visualization of Decision Boundary  
For visualization purposes, only two standardized features were used. Decision regions were plotted using a mesh grid to show how the SVM classifier separates the classes.
