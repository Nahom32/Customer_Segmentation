import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def training_pipeline():
    # Load the dataset
    df = pd.read_csv("data/Loan_Default.csv")

    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
    # Fill numerical missing values with median
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Fill categorical missing values with mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    # Encode categorical variables
    cat_columns = df.select_dtypes(include=["object"]).columns
    label_encoders = {}
    for col in cat_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Store the encoder

    # Save label encoders
    with open("label_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)
    df.drop(columns=["ID"], inplace=True)

    X = df.drop(columns=["Status"])
    y = df["Status"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    with open("loan_model.pkl", "wb") as f:
        pickle.dump(model, f)