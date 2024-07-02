import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('trained_model.sav', 'rb'))

def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def main():
    st.title('Diabetes Prediction Web App')

    Pregnancies = st.text_input('Number of Pregnancies', '0')
    Glucose = st.text_input('Glucose Level', '85')
    BloodPressure = st.text_input('Blood Pressure value', '80')
    SkinThickness = st.text_input('Skin Thickness value', '20')
    Insulin = st.text_input('Insulin Level', '85')
    BMI = st.text_input('BMI value', '22.0')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value', '0.5')
    Age = st.text_input('Age of the Person', '30')

    diagnosis = ''

    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.success(diagnosis)

if __name__ == '__main__':
    main()
