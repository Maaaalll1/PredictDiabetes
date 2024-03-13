# app.py

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load diabetes dataset
@st.cache
def load_data():
    return pd.read_csv("diabetes.csv")

# Preprocess data
def preprocess_data(data):
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

# Train KNN model
def train_model(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    return knn

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Model Accuracy:", accuracy)
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))
    st.write("Classification Report:")
    st.write(classification_report(y_test, y_pred))

def main():
    st.title("Diabetes Prediction App")
    st.write("This app demonstrates a simple diabetes prediction model using K-Nearest Neighbors.")
    
    # Load data
    data = load_data()
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    st.subheader("Model Evaluation")
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
