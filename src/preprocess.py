import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess_data():
    print("Loading raw data...")

    df = pd.read_csv('../data/raw/survey lung cancer.csv')

    print("Mapping categorical variables...")

    df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})
    df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

    X = df.drop(columns=['LUNG_CANCER'])
    y = df['LUNG_CANCER']

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Scaling the features...")
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    train_data = X_train_df.copy()
    train_data['LUNG_CANCER'] = y_train.values
    
    test_data = X_test_df.copy()
    test_data['LUNG_CANCER'] = y_test.values

    print("Saving processed data and scaler...")
    os.makedirs('../data/processed', exist_ok=True)
    os.makedirs('../models', exist_ok=True)

    train_data.to_csv('../data/processed/train_cleaned.csv', index=False)
    test_data.to_csv('../data/processed/test_cleaned.csv', index=False)

    joblib.dump(scaler, '../models/scaler.pkl')

    print("Preprocessing complete!")

if __name__ == "__main__":
    preprocess_data()