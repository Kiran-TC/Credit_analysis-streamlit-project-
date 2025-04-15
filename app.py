import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Title
st.title("Credit Card Fraud Detection")
st.write("Using Logistic Regression on Credit Card Transactions")

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("creditcard.csv")
    return data

# Preprocess data
def preprocess(data):
    X = data.drop("Class", axis=1)
    y = data["Class"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train model
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

# Main app
st.subheader("Dataset Preview")
data = load_data()
st.write(data.head())

if st.button("Train Model"):
    X_train, X_test, y_train, y_test = preprocess(data)
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("Model Performance")
    st.write("**Accuracy:**", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
    st.write("**Confusion Matrix:**")
    st.write(confusion_matrix(y_test, y_pred))
    st.write("**Classification Report:**")
    st.text(classification_report(y_test, y_pred))
