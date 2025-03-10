import streamlit as st
import joblib
import os
import numpy as np


@st.cache_data
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file),'rb'))
    return loaded_model


def main():
    st.subheader("Loan Approval Application")    
    person_age = st.number_input("Loan applicant age")
    if person_age < 0:
         st.warning("age must be equal to or above 0")
    person_income = st.number_input("Loan applicant income")
    if person_income < 0:
        st.warning("income must be equal to or above 0")
    person_emp_exp = st.number_input("Loan applicant employement experience")
    if person_emp_exp < 0:
        st.warning("employement experience must be equal to or above 0")
    loan_amount = st.number_input("loan amount requested")
    if loan_amount < 1:
        st.warning("loan amount must be equal to or above 1")
    loan_int_rate = st.number_input("Loan interest rate")
    if loan_int_rate < 0:
        st.warning("loan interest rate must be equal to or above 0")
    cb_cred_hist_length = st.number_input("Length of credit history in years")
    if cb_cred_hist_length < 0:
        st.warning("Length of credit history must be equal to or above 0")
    credit_score = st.number_input("Loan applicant credit score")
    if credit_score < 0:
        st.warning("credit score must be equal to or above 0")
    
    # Opsi radio button
    gender_options = ['Male', 'Female']
    education_options = ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']
    home_ownership_options = ['Own', 'Mortgage', 'Rent', 'Other']
    loan_intent_options = ['Personal', 'Education', 'Venture', 'Homeimprovement', 'Medical', 'Debtconsolidation']
    prev_loan_def_options = ['Yes', 'No']

    # Input dari user
    person_gender = st.radio("Loan applicant gender", gender_options)
    person_education = st.radio("Loan applicant education", education_options)
    person_home_ownership = st.radio("Loan applicant home ownership", home_ownership_options)
    loan_intent = st.radio("Loan intention", loan_intent_options)
    prev_loan_def_on_file = st.radio("Indicator of previous loan defaults", prev_loan_def_options)

    # Mengonversi pilihan user menjadi indeks
    person_gender_idx = gender_options.index(person_gender)
    person_education_idx = education_options.index(person_education)
    person_home_ownership_idx = home_ownership_options.index(person_home_ownership)
    loan_intent_idx = loan_intent_options.index(loan_intent)
    prev_loan_def_on_file_idx = prev_loan_def_options.index(prev_loan_def_on_file)

    input_feature = np.array([
    person_age, person_gender_idx, person_education_idx, person_income, 
    person_emp_exp, person_home_ownership_idx, loan_amount, 
    loan_intent_idx, loan_int_rate, cb_cred_hist_length, 
    credit_score, prev_loan_def_on_file_idx])

    if st.button("Predict"):
        st.write(input_feature.reshape(1,-1))

        model = load_model("lgbm_model.pkl")

        prediction = model.predict(input_feature.reshape(1,-1))
        st.write("Loan applicant approval status is: ")
        if prediction == 1:
            st.markdown("<div style='text-align: center; font-size: 24px;background-color: green;'>Approved</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align: center; font-size: 24px;background-color: red;'>rejected</div>", unsafe_allow_html=True)


if __name__ == '__main__':
    main()