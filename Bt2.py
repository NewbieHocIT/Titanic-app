import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
df.dropna(inplace=True)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Split data
X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Display dataset sizes
train_size = X_train.shape[0]
valid_size = X_valid.shape[0]
test_size = X_test.shape[0]

# Model training & saving
model_path = "random_forest_model.pkl"
metrics_path = "model_metrics.txt"

if not os.path.exists(model_path):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    cv_accuracy = np.mean(scores)
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save metrics
    with open(metrics_path, 'w') as f:
        f.write(f"cv_accuracy:{cv_accuracy}\n")
        f.write(f"test_accuracy:{test_acc}\n")
else:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load metrics
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            lines = f.readlines()
            cv_accuracy = float(lines[0].split(":")[1])
            test_acc = float(lines[1].split(":")[1])
    else:
        cv_accuracy, test_acc = None, None

# Streamlit UI
st.title("Titanic Survival Prediction")
st.write("## Model Performance")
if cv_accuracy is not None and test_acc is not None:
    st.write(f"Cross-validation Accuracy: {cv_accuracy:.4f}")
    st.write(f"Test Accuracy: {test_acc:.4f}")
else:
    st.write("Model metrics not available.")

# Display dataset sizes
st.write("## Dataset Sizes")
st.write(f"Train Set: {train_size} samples")
st.write(f"Validation Set: {valid_size} samples")
st.write(f"Test Set: {test_size} samples")

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

# Make Prediction
if st.sidebar.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    result = "Survived" if prediction == 1 else "Did Not Survive"
    st.write(f"### Prediction: {result}")
