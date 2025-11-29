import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and encoders
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💰 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs
st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 17, 90, 30)

workclass = st.sidebar.selectbox("Workclass", encoders['workclass'].classes_)
education = st.sidebar.selectbox("Education Level", encoders['education'].classes_)
marital_status = st.sidebar.selectbox("Marital Status", encoders['marital-status'].classes_)
occupation = st.sidebar.selectbox("Occupation", encoders['occupation'].classes_)
gender = st.sidebar.selectbox("Gender", encoders['gender'].classes_)
hours_per_week = st.sidebar.slider("Hours per week", 1, 99, 40)

# Build input DataFrame
input_data = {
    'age': [age],
    'workclass': [workclass],
    'education': [education],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'gender': [gender],
    'hours-per-week': [hours_per_week]
}

input_df = pd.DataFrame(input_data)

st.write("### Input Data")
st.write(input_df)

# Preprocess input
# Encode categorical variables using the loaded encoders
for col, le in encoders.items():
    if col in input_df.columns:
        input_df[col] = le.transform(input_df[col])

# Predict button
if st.button("Predict Salary Class"):
    try:
        prediction = model.predict(input_df)
        st.success(f"Prediction: {prediction[0]}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Batch prediction
st.markdown("---")
st.markdown("### 📊 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:", batch_data.head())
        
        # Preprocess batch data
        # Ensure columns match
        required_cols = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'gender', 'hours-per-week']
        if all(col in batch_data.columns for col in required_cols):
            # Encode
            processed_batch = batch_data.copy()
            for col, le in encoders.items():
                if col in processed_batch.columns:
                    # Handle unseen labels if necessary, or just transform
                    # For simplicity, we assume valid data or drop invalid
                    processed_batch = processed_batch[processed_batch[col].isin(le.classes_)]
                    processed_batch[col] = le.transform(processed_batch[col])
            
            batch_preds = model.predict(processed_batch[required_cols])
            batch_data = batch_data.loc[processed_batch.index] # Align with filtered data
            batch_data['PredictedClass'] = batch_preds
            st.write("Predictions:")
            st.write(batch_data.head())
            csv = batch_data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
        else:
            st.error(f"Uploaded CSV must contain the following columns: {required_cols}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
