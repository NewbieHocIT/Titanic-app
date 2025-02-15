import streamlit as st
import pandas as pd
import numpy as np
import joblib
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

# Dataset sizes
train_size, valid_size, test_size = len(X_train), len(X_valid), len(X_test)

# Model training & saving
model_path = "random_forest_model.pkl"

if not os.path.exists(model_path):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    cv_accuracy = np.mean(scores)

    # Save model
    joblib.dump(model, model_path)

    # Save metrics
    model_metrics = {"cv_accuracy": cv_accuracy, "test_accuracy": test_acc}
else:
    # Load model
    try:
        model = joblib.load(model_path)
        model_metrics = {"cv_accuracy": None, "test_accuracy": None}
    except:
        st.error("🚨 Error loading the model!")
        st.stop()

# Streamlit UI
st.title("🚢 Titanic Survival Prediction")
st.subheader("💡 Model Performance")
if model_metrics["cv_accuracy"] is not None:
    st.write(f"✅ **Cross-validation Accuracy:** `{model_metrics['cv_accuracy']:.4f}`")
    st.write(f"✅ **Test Accuracy:** `{model_metrics['test_accuracy']:.4f}`")
else:
    st.write("⚠️ Model metrics not available.")

# Display dataset sizes
st.subheader("📊 Dataset Sizes")
st.write(f"📌 **Train Set:** `{train_size}` samples")
st.write(f"📌 **Validation Set:** `{valid_size}` samples")
st.write(f"📌 **Test Set:** `{test_size}` samples")

# Sidebar for User Input
st.sidebar.header("📝 Enter Passenger Details")
pclass = st.sidebar.selectbox("🔹 Pclass", [1, 2, 3])
sex = st.sidebar.selectbox("🔹 Sex", ["Male", "Female"])
age = st.sidebar.number_input("🔹 Age", min_value=0, max_value=100, value=30)
sibsp = st.sidebar.number_input("🔹 SibSp", min_value=0, max_value=10, value=0)
parch = st.sidebar.number_input("🔹 Parch", min_value=0, max_value=10, value=0)
fare = st.sidebar.number_input("🔹 Fare", min_value=0.0, max_value=500.0, value=32.0)

# Convert inputs
sex = 0 if sex == "Male" else 1
input_data = np.array([[pclass, sex, age, sibsp, parch, fare]], dtype=np.float64)

# Make Prediction
if st.sidebar.button("🚀 Predict Survival"):
    try:
        prediction = model.predict(input_data)[0]
        result = "🟢 Survived" if prediction == 1 else "🔴 Did Not Survive"
        st.subheader(f"🎯 Prediction: {result}")
    except Exception as e:
        st.error(f"🚨 Prediction Error: {e}")
