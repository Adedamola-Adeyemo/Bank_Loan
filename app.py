import streamlit as st
import joblib
import pandas as pd

file = open('loan.joblib','rb')
model = joblib.load(file)

st.title('Bank Loan Default Prediction')
st.subheader('Which customer will default if given a loan')
st.write('Customers not paying back loan is a major issue faced in Nigerian banks. Giving loans to the right category of people is very important to the financial progress of any banking institution. This is a machine learning (ML) model with 97% accuracy that predicts if a customer will default on a loan or not, it also displays the probability for defaulting.')

html_temp = """
    <div style ='background-color: orange; padding:10px'>
    <h2> ML Based Web App </h2>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)
st.write('Here is a sample data for some customers')
sample = pd.read_csv('sample.csv')
st.write(sample.sample(frac=1))
st.write('')
st.write('Please provide the following customer details')
student = st.selectbox('Is the customer a student?',('No','Yes'))
balance = st.text_input('What is the balance in the bank account of the customer?')
income = st.text_input('What is the income of the customer?')

features = {'student':student,
'balance':balance,
'income':income,
}
if st.button('Submit'):
    data = pd.DataFrame(features,index=[0,1])
    st.write(data)

    prediction = model.predict(data)
    proba = model.predict_proba(data)[1]

    if prediction[0] == 0:
        st.success('This customer will pay back the loan')
    else:
        st.error('The customer will default on the loan')

    proba_df = pd.DataFrame(proba,columns=['Probability'],index=['Default : No','Default : Yes'])
    #proba_df.plot(kind='barh')
    st.bar_chart(proba_df)