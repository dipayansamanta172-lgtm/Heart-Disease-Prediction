# Heart Disease Prediction using Machine Learning

Predict heart disease status using machine learning by analyzing patient health and lifestyle features.

[![Python](https://img.shields.io/badge/Python-3-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4C72B0)](https://seaborn.pydata.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Model-189A46)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-Open%20Source-green)](https://github.com/dipayansamanta172-lgtm/Heart-Disease-Prediction)

## Download

[![Download Project](https://img.shields.io/badge/Download-Project-blue)](https://github.com/dipayansamanta172-lgtm/Heart-Disease-Prediction/archive/refs/heads/main.zip)
[![Download Dataset](https://img.shields.io/badge/Download-Dataset-green)](https://github.com/dipayansamanta172-lgtm/Heart-Disease-Prediction/raw/main/MiniProject_LogisticRegression/heart.csv)

## Overview

This project compares three machine learning classification models for heart disease prediction: Logistic Regression, Random Forest, and XGBoost.

The dataset is processed using data cleaning, categorical feature encoding, feature scaling, SMOTE-based class balancing, model training, and evaluation. Each model folder contains its Python implementation, dataset, generated graphs, and results file.

## Features

- Data cleaning and missing-value handling
- Categorical feature encoding
- Feature scaling using StandardScaler
- Class balancing using SMOTE
- Exploratory data analysis and visualization
- Logistic Regression, Random Forest, and XGBoost models
- Confusion matrix and ROC curve generation
- Model performance evaluation
- Feature importance visualization

## Technologies Used

- Python 3
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Imbalanced-Learn
- XGBoost

## Dataset

**heart.csv** contains the patient records used for heart disease classification.

**Target Variable:** Heart Disease Status

The same dataset is included inside each model folder so that every implementation can be executed independently.

## Repository Structure

```text
.
├── MiniProject_LogisticRegression/
│   ├── graphs/
│   │   ├── figure1_eda_collage.png
│   │   ├── figure2_class_distribution_before_smote.png
│   │   ├── figure3_missing_values.png
│   │   ├── figure4_confusion_matrix - LR.png
│   │   ├── figure5_roc_curve - LR.png
│   │   ├── figure6_model_performance_comparison.png
│   │   ├── figure7_correlation_heatmap.png
│   │   └── figure8_feature_importance.png
│   ├── output/
│   │   └── results.txt
│   ├── heart.csv
│   └── logistic_regression.py
│
├── MiniProject_RandomForest/
│   ├── graphs/
│   │   ├── figure1_eda_collage.png
│   │   ├── figure2_class_distribution_before_smote.png
│   │   ├── figure3_missing_values.png
│   │   ├── figure4_confusion_matrix - RF.png
│   │   ├── figure5_roc_curve.png
│   │   ├── figure6_model_performance_comparison.png
│   │   ├── figure7_correlation_heatmap.png
│   │   └── figure8_feature_importance.png
│   ├── output/
│   │   └── results.txt
│   ├── heart.csv
│   └── random_forest.py
│
├── MiniProject_xgboost/
│   ├── graphs/
│   │   ├── figure1_eda_collage.png
│   │   ├── figure2_class_distribution.png
│   │   ├── figure3_missing_values.png
│   │   ├── figure4_confusion_matrix XGBoost.png
│   │   ├── figure5_roc_curve.png
│   │   ├── figure6_model_performance.png
│   │   ├── figure7_correlation_heatmap.png
│   │   └── figure8_feature_importance.png
│   ├── output/
│   │   └── results.txt
│   ├── heart.csv
│   └── xgboost_model.py
│
├── LICENSE
└── README.md
```

## How It Works

1. **Dataset Loading** - Reads `heart.csv`.
2. **Data Cleaning** - Checks missing values and duplicate records and handles missing data.
3. **Feature Preparation** - Encodes categorical features and prepares the target variable.
4. **Train-Test Split** - Splits the dataset into training and testing sets using an 80:20 ratio.
5. **Feature Scaling** - Applies StandardScaler to the feature data.
6. **SMOTE Balancing** - Balances the training data using SMOTE.
7. **Model Training** - Trains the selected machine learning classifier.
8. **Evaluation** - Calculates Accuracy, Precision, Recall, F1 Score, and ROC-AUC Score.
9. **Visualization** - Generates EDA charts, confusion matrices, ROC curves, correlation heatmaps, and feature-importance graphs.
10. **Output Generation** - Saves model evaluation results in `output/results.txt`.

## Running the Projects

Run each model independently from inside its corresponding folder.

### Logistic Regression

**Windows:**
```text
cd MiniProject_LogisticRegression
python logistic_regression.py
```

**Linux/macOS:**
```text
cd MiniProject_LogisticRegression
python3 logistic_regression.py
```

### Random Forest

**Windows:**
```text
cd MiniProject_RandomForest
python random_forest.py
```

**Linux/macOS:**
```text
cd MiniProject_RandomForest
python3 random_forest.py
```

### XGBoost

**Windows:**
```text
cd MiniProject_xgboost
python xgboost_model.py
```

**Linux/macOS:**
```text
cd MiniProject_xgboost
python3 xgboost_model.py
```

## Installation

### Prerequisites

[![Download Python](https://img.shields.io/badge/Download-Python-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Download VS Code](https://img.shields.io/badge/Download-VS%20Code-blue?logo=visual-studio-code&logoColor=white)](https://code.visualstudio.com/)

### Install Required Libraries

```text
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn xgboost
```

## Output Files

Each model folder generates:

**graphs/** - Visualization files including EDA charts, class distribution, missing-value analysis, confusion matrix, ROC curve, model performance, correlation heatmap, and feature importance.

**output/results.txt** - Model evaluation results containing:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

## Results

### Logistic Regression

- Accuracy: **51.65%**
- Precision: **18.88%**
- Recall: **43.00%**
- F1 Score: **26.24%**
- ROC-AUC: **0.4802**

### Random Forest

- Accuracy: **79.15%**
- Precision: **20.69%**
- Recall: **1.50%**
- F1 Score: **2.80%**
- ROC-AUC: **0.4931**

### XGBoost

- Accuracy: **79.15%**
- Precision: **18.52%**
- Recall: **1.25%**
- F1 Score: **2.34%**
- ROC-AUC: **0.4864**

## Future Improvements

- Hyperparameter tuning
- Cross-validation for more robust evaluation
- Improved handling of class imbalance
- Additional feature engineering
- Testing on larger and more diverse datasets
- Deployment as a web or API-based prediction system

---

**Dataset:** heart.csv  
**Repository:** [Heart Disease Prediction](https://github.com/dipayansamanta172-lgtm/Heart-Disease-Prediction)  
**Author:** Dipayan Samanta  
**License:** Open source
