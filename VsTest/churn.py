import streamlit as st
import numpy as np
import pandas as pd 
import plotly.express as px 
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
from plotly.subplots import make_subplots
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Importing data
data=pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df=pd.read_csv('churn_data.csv')

# Designing page
st.set_page_config(page_title='CUSTOMER CHURN PREDICTION', layout='wide', page_icon="random")
st.title("CUSTOMER CHURN PREDICTION")
st.divider()
st.sidebar.divider()
selected_tab = st.sidebar.radio("OVERVIEW", ['Introduction','Dataset','EDA','Model','Conclusion','Prediction/Application'])
st.sidebar.divider()

if selected_tab == 'Introduction':
    st.markdown("In developed countries, telecommunications is a critical industry where customer attrition, or churn, poses a significant challenge, particularly in mature markets. Churn can be accidental, due to changes in a customer's circumstances, or intentional, when customers switch to competitors. Companies prioritize reducing voluntary churn, which results from controllable factors like billing or customer support. Retention is crucial for subscription-based models, as losing customers leads to economic and reputational harm. This study focuses on identifying the main causes of churn among fixed telephony subscribers in a telecom company.")
    c1,c2 = st.columns(2)
    with c1:
        st.image('chn.jpg')
    with c2:
        st.subheader('Business Understanding Churn Prediction')
        st.write('Identifying customers who are likely to cancel their contracts soon.')
        st.write('If the company can do that, it can handle users before churn.The target variable that we want to predict is categorical and has only two possible outcomes: churn or not churn (Binary Classification).We also would like to understand why the model thinks our customers churn, and for that, we need to be able to interpret the model’s predictions.')
    st.divider()
    

elif selected_tab == 'Dataset':
    st.subheader('Data overview')
    st.divider()
    st.subheader('Dataset')
    data
    st.divider()
    st.subheader('According to the description, this dataset has the following information:')
    st.write('Services of the customers: phone; multiple lines; internet; tech support and extra services such as online security, backup, device protection, and TV streaming.')
    st.write('Account information: how long they have been clients, type of contract, type of payment method.')
    st.write('Charges: how much the client was charged in the past month and in total.')
    st.write('Demographic information: gender, age, and whether they have dependents or a partner')
    st.write('Churn: yes/no, whether the customer left the company within the past month.')
    st.divider()


elif selected_tab == 'EDA':
    st.subheader('Exploratory Data Analysis')
    st.divider()
    # Assuming df is already loaded as your DataFrame

    # 1. Distribution of Churn
    st.subheader("Distribution of Churn")
    churn_distribution = df['Churn'].value_counts().reset_index()
    churn_distribution.columns = ['Churn', 'Count']
    fig1 = px.bar(churn_distribution, x='Churn', y='Count', title='Churn Distribution')
    st.plotly_chart(fig1)

    # 2. Customer Demographics
    st.subheader("Customer Demographics and Churn")

    # Gender
    gender_churn = df.groupby(['gender', 'Churn']).size().reset_index(name='Count')
    fig2 = px.bar(gender_churn, x='gender', y='Count', color='Churn', barmode='group', title='Churn by Gender')
    st.plotly_chart(fig2)

    # Senior Citizen
    senior_churn = df.groupby(['SeniorCitizen', 'Churn']).size().reset_index(name='Count')
    fig3 = px.bar(senior_churn, x='SeniorCitizen', y='Count', color='Churn', barmode='group', title='Churn by Senior Citizen')
    st.plotly_chart(fig3)

    # Partner and Dependents
    partner_dependents_churn = df.groupby(['Partner', 'Dependents', 'Churn']).size().reset_index(name='Count')
    fig4 = px.bar(partner_dependents_churn, x='Partner', y='Count', color='Churn', facet_col='Dependents', barmode='group', title='Churn by Partner and Dependents')
    st.plotly_chart(fig4)

    # 3. Tenure Analysis
    st.subheader("Tenure Analysis and Churn")
    fig5 = px.histogram(df, x='tenure', color='Churn', nbins=50, title='Tenure Distribution by Churn')
    st.plotly_chart(fig5)

    # 4. Services Subscribed
    st.subheader("Services Subscribed and Churn")

    # Phone Service
    phone_churn = df.groupby(['PhoneService', 'Churn']).size().reset_index(name='Count')
    fig6 = px.bar(phone_churn, x='PhoneService', y='Count', color='Churn', barmode='group', title='Churn by Phone Service')
    st.plotly_chart(fig6)

    # Internet Service
    internet_churn = df.groupby(['InternetService', 'Churn']).size().reset_index(name='Count')
    fig7 = px.bar(internet_churn, x='InternetService', y='Count', color='Churn', barmode='group', title='Churn by Internet Service')
    st.plotly_chart(fig7)

    # Multiple Lines
    multiplelines_churn = df.groupby(['MultipleLines', 'Churn']).size().reset_index(name='Count')
    fig8 = px.bar(multiplelines_churn, x='MultipleLines', y='Count', color='Churn', barmode='group', title='Churn by Multiple Lines')
    st.plotly_chart(fig8)

    # 5. Contract Type
    st.subheader("Contract Type and Churn")
    contract_churn = df.groupby(['Contract', 'Churn']).size().reset_index(name='Count')
    fig9 = px.bar(contract_churn, x='Contract', y='Count', color='Churn', barmode='group', title='Churn by Contract Type')
    st.plotly_chart(fig9)

    # 6. Billing Preferences
    st.subheader("Billing Preferences and Churn")

    # Paperless Billing
    paperlessbilling_churn = df.groupby(['PaperlessBilling', 'Churn']).size().reset_index(name='Count')
    fig10 = px.bar(paperlessbilling_churn, x='PaperlessBilling', y='Count', color='Churn', barmode='group', title='Churn by Paperless Billing')
    st.plotly_chart(fig10)

    # Payment Method
    paymentmethod_churn = df.groupby(['PaymentMethod', 'Churn']).size().reset_index(name='Count')
    fig11 = px.bar(paymentmethod_churn, x='PaymentMethod', y='Count', color='Churn', barmode='group', title='Churn by Payment Method')
    st.plotly_chart(fig11)

    # 7. Charges Analysis
    st.subheader("Charges Analysis and Churn")

    # Monthly Charges
    fig12 = px.box(df, x='Churn', y='MonthlyCharges', title='Monthly Charges by Churn')
    st.plotly_chart(fig12)

    # Total Charges
    fig13 = px.box(df, x='Churn', y='TotalCharges', title='Total Charges by Churn')
    st.plotly_chart(fig13)

    # 8. Correlation Matrix
    st.subheader("Correlation Matrix")
    plt.figure(figsize=(12, 8))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    st.pyplot(plt)

    # 9. Customer Service Impact
    st.subheader("Customer Service Impact and Churn")

    # Tech Support
    techsupport_churn = df.groupby(['TechSupport', 'Churn']).size().reset_index(name='Count')
    fig14 = px.bar(techsupport_churn, x='TechSupport', y='Count', color='Churn', barmode='group', title='Churn by Tech Support')
    st.plotly_chart(fig14)

    # Online Security
    onlinesecurity_churn = df.groupby(['OnlineSecurity', 'Churn']).size().reset_index(name='Count')
    fig15 = px.bar(onlinesecurity_churn, x='OnlineSecurity', y='Count', color='Churn', barmode='group', title='Churn by Online Security')
    st.plotly_chart(fig15)

    # Device Protection
    deviceprotection_churn = df.groupby(['DeviceProtection', 'Churn']).size().reset_index(name='Count')
    fig16 = px.bar(deviceprotection_churn, x='DeviceProtection', y='Count', color='Churn', barmode='group', title='Churn by Device Protection')
    st.plotly_chart(fig16)

    # 10. Churn by Tenure and Monthly Charges
    st.subheader("Churn by Tenure and Monthly Charges")
    fig17 = px.scatter(df, x='tenure', y='MonthlyCharges', color='Churn', title='Churn by Tenure and Monthly Charges')
    st.plotly_chart(fig17)
    st.divider()

elif selected_tab == 'Model':
    st.subheader('Business Understanding Churn Prediction')
    
    st.divider()

elif selected_tab == 'Conclusion':
    st.subheader('Business Understanding Churn Prediction')
    st.divider()

elif selected_tab == 'Prediction/Application':
    st.subheader('Business Understanding Churn Prediction')
    st.divider()
