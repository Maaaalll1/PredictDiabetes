# main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import tree
import graphviz
import matplotlib.image as mpimg
from IPython.display import Image
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import keras
from keras.models import Sequential
from keras.layers import Dense


%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

def load_data():
    # Load the dataset
    data = pd.read_csv("diabetes.csv")
    
    # Display basic information about the dataset
    print("Dataset shape:", data.shape)
    print("\nFirst few rows of the dataset:\n", data.head())
    
    return data

def preprocess_data(data):
    # Separate features and target variable
    X = data.drop('target_column_name', axis=1)  # Adjust 'target_column_name' to the actual name of the target column
    y = data['target_column_name']
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_model(X_train, y_train):
    # Initialize and train the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    return knn

def evaluate_model(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("\nModel accuracy:", accuracy)
    
    # Display confusion matrix and classification report
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

def main():
    # Load the dataset
    data = load_data()
    
    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
