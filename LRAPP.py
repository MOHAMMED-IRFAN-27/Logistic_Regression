import streamlit as st
import pandas as pd
import joblib

# Load the trained logistic regression model
model = joblib.load("titanic_logreg_model.pkl")

st.title("Titanic Survival Prediction ")

st.write("Enter passenger details to predict survival:")

# User inputs
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Siblings/Spouses aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=32.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])
# Convert categorical inputs to numeric / one-hot
sex_numeric = 1 if sex == "male" else 0
embarked_C = 1 if embarked == "C" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

input_data = pd.DataFrame([[pclass, sex_numeric, age, sibsp, parch, fare, embarked_C, embarked_Q, embarked_S]],
                          columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_C", "Embarked_Q", "Embarked_S"])


# Convert categorical 'Sex' to numeric (assuming model was trained this way)
sex_numeric = 1 if sex == "male" else 0

# Prepare input for model
input_data = pd.DataFrame([[pclass, sex_numeric, age, sibsp, parch, fare]],
                          columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"])

# Prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # Probability of survival
    
    if prediction == 1:
        st.success(f"The passenger is likely to **survive** with probability {probability:.2f}")
    else:
        st.error(f" The passenger is likely **not to survive** with probability {1 - probability:.2f}")
