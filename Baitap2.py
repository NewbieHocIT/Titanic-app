import os
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("C:/TraThanhTri/PYthon/TriTraThanh/MLvsPython/processed_data.csv")

# Split data
X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Check if model exists
MODEL_PATH = "random_forest_model.pkl"
if not os.path.exists(MODEL_PATH):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    
    # Save accuracy metrics
    with open("model_metrics.txt", "w") as f:
        f.write(f"cv_accuracy: {np.mean(scores)}\n")
        f.write(f"test_accuracy: {test_acc}\n")
else:
    with open("model_metrics.txt", "r") as f:
        metrics = f.readlines()
    cv_accuracy = float(metrics[0].split(": ")[1])
    test_acc = float(metrics[1].split(": ")[1])

# Streamlit UI
st.title("Titanic Survival Prediction")
st.write("## Model Performance")
st.write(f"Training Set Size: {X_train.shape[0]}")
st.write(f"Validation Set Size: {X_valid.shape[0]}")
st.write(f"Test Set Size: {X_test.shape[0]}")
st.write(f"Cross-validation Accuracy: {cv_accuracy:.4f}")
st.write(f"Test Accuracy: {test_acc:.4f}")

# User Input for Prediction
st.sidebar.header("Enter Passenger Details")
pclass = st.sidebar.selectbox("Pclass", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.sidebar.number_input("SibSp", min_value=0, max_value=10, value=0)
parch = st.sidebar.number_input("Parch", min_value=0, max_value=10, value=0)
fare = st.sidebar.number_input("Fare", min_value=0.0, max_value=500.0, value=32.0)

# Convert inputs
sex = 0 if sex == "Male" else 1
input_data = np.array([[pclass, sex, age, sibsp, parch, fare]])

# Load Model and Predict
if st.sidebar.button("Predict Survival"):
    try:
        with open(MODEL_PATH, "rb") as f:
            loaded_model = pickle.load(f)
        prediction = loaded_model.predict(input_data)[0]
        result = "Survived" if prediction == 1 else "Did Not Survive"
        st.write(f"### Prediction: {result}")
    except Exception as e:
        st.error(f"Error loading model: {e}")
