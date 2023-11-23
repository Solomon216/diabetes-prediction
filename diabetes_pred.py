import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st

diabetes = pd.read_csv('diabetes.csv')

features = ['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']
label = 'Diabetic'
X, y = diabetes[features].values, diabetes[label].values

for n in range(0,4):
    print("Patient", str(n+1), "\n  Features:",list(X[n]), "\n  Label:", y[n])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

reg = 0.01

model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

st.title("Diabetes Prediction App")

st.subheader("Enter patient information:")
preg=st.number_input("Pregnancies :",min_value=0)
pg=st.number_input("Plasma GLucose :",min_value=0)
dbp = st.number_input("Diastolic Blood Presssure :",min_value=0)
tt=st.number_input("Tricep Thickness :",min_value=0)
si=st.number_input("Serum Insulin :",min_value=0)
bmi = st.number_input("BMI :",min_value=0)
dp=st.number_input("Daibetes Pedigree :",min_value=0)
age=st.number_input("Age :",min_value=0)

X_new = np.array([[preg,pg,dbp,tt,si,bmi,dp,age]])

pred = model.predict(X_new)

if st.button("Predict"):
    st.subheader("Prediction:")
    if pred[0]==1:
        st.write("You are diabetic!")
    else:
        st.write ("You are not daibetic")
