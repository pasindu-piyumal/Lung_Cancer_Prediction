# 🫁 Lung Cancer Risk Prediction Pipeline

An end-to-end Machine Learning pipeline predicting lung cancer risk from patient survey data. Built with scikit-learn, featuring automated preprocessing and class imbalance handling.

## 📊 Dataset Overview
The project uses the **Survey Lung Cancer Dataset**.
* **Total Patients:** 309
* **Features:** 15 variables including Age, Gender, Smoking, Anxiety, Wheezing, etc.
* **Target:** `LUNG_CANCER` (YES/NO)
* **Challenge Addressed:** The dataset is highly imbalanced (270 Positive vs 39 Negative cases). This pipeline handles the imbalance using `class_weight='balanced'` within a Random Forest Classifier to ensure rare negative cases are not ignored by the model.

## 📂 Repository Structure
```text
lung_cancer_prediction/
├── data/
│   ├── raw/survey lung cancer.csv       
│   └── processed/                       
├── notebooks/
│   ├── 01_data_exploration.ipynb        
│   └── 02_model_training.ipynb          
├── models/
│   ├── lung_cancer_rf_model.pkl         
│   └── scaler.pkl                       
├── src/                                  
│   ├── preprocess.py                    
│   └── predict.py                       
├── requirements.txt                     
└── README.md