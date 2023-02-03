import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
from src import transform_response, cols_order
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from PIL import Image


class CONFIG():
    current_path = os.getcwd()
    dataset_path = os.path.join(os.getcwd(), 'dataset')
    info_path = os.path.join(os.getcwd(), 'info')
    models_path = os.path.join(os.getcwd(), 'models')
    notebooks_path = os.path.join(os.getcwd(), 'notebooks')
    model_name = 'xgb_model_2000_est'
    encoder_name = 'ordinal_encoder'
    icons_path = os.path.join(os.getcwd(), 'icons')
    logo_path = os.path.join(icons_path, 'logo.png')
    logo_img = Image.open(logo_path)
    n_estimators = 2000
    num_features = 16
    num_output = 1
    features_description_path = os.path.join(
        os.getcwd(), 'files/feature meanings of the dataset.csv')
    output_transform_dict = {
        0: 'Good',
        1: 'Poor',
        2: 'Standard'
    }

    @st.cache(allow_output_mutation=True)
    def get_model(self):
        model = xgb.XGBClassifier(n_estimators=self.n_estimators)
        model.load_model(os.path.join(
            self.models_path, self.model_name + '.json'))
        return model

    @st.cache(allow_output_mutation=True)
    def unzip_load(self, path):
        return pickle.load(open(path, 'rb'))

    @st.cache(allow_output_mutation=True)
    def get_encoder(self):
        return self.unzip_load(os.path.join(os.getcwd(), config.encoder_name + './pkl'))

    @st.cache(allow_output_mutation=True)
    def get_feature_description_df(self):
        return pd.read_csv(self.features_description_path)


config = CONFIG()
st.set_page_config(page_title='Credit Score App',  page_icon=config.logo_img, layout='wide',
                   initial_sidebar_state='auto', menu_items={
                       'Report a bug': 'https://github.com/Somil7Jain/Credit-Score-Classification/issues',
                       'About': '''
                        This app was made by **Somil Jain** for predicting credit score based on some inputs given by user.
                        visit github profile (https://github.com/Somil7Jain/Credit-Score-Classification) for more info.
                        '''
                   })

st.title('Credit Score Classification')
st.caption('Made by Somil Jain')

st.code('''
Click the sidebar icon to open the sidebar. You can fill out the given form
to predict the credit score of an individual user. (this data is not saved).
            ''')

model = config.get_model()
df = config.get_feature_description_df()

with st.expander("See explanation of features for prediction of credit score (All money related terms are in USD)"):
    st.dataframe(df)

annual_income_default = 45000.00
monthly_inhand_salary_default = 3750.00
num_accounts_default = 3
num_credit_cards_default = 2
interest_rate_default = 14.0
num_loans_default = 3
delay_from_due_date_default = 22
num_delayed_payments_default = 5
changed_credit_limit_default = 11.00
num_credit_inqueries_default = 6
credit_mix_default = 'Standard'
outstanding_debt_default = 1430.00
credit_history_age_default = 222
payment_min_amount_default = 'Yes'
total_emi_per_month_default = 350.00
amount_invested_monthly_default = 350.00

with st.sidebar:
    st.header('Credit Score Form')
    st.caption('All money related input values are in USD (US dollars)')
    annual_income = st.number_input(
        'Please enter your Annual Income.', min_value=0.00, max_value=300000.00, value=annual_income_default)
    monthly_inhand_salary = st.number_input(
        'Please enter your monthly base salary.', min_value=0.00, max_value=25000.00, value=monthly_inhand_salary_default)

    num_accounts = st.number_input('Please enter number of your bank accounts.',
                                   min_value=0, max_value=20, step=1, value=num_accounts_default)
    num_credit_cards = st.number_input('Please enter number of your credit cards.',
                                       min_value=0, max_value=12, step=1, value=num_credit_cards_default)
    interest_rate = st.number_input('Please enter your average interest rate on credit card',
                                    min_value=0.00, max_value=35.00, step=0.01, value=interest_rate_default)
    num_loans = st.number_input('Please enter number of your loans.',
                                min_value=0, max_value=10, step=1, value=num_loans_default)
    delay_from_due_date = st.number_input('Please enter the average number of days delayed from the payment date.',
                                          min_value=0, max_value=65, step=1, value=delay_from_due_date_default)
    num_delayed_payments = st.number_input('Please enter the number of delayed payments.',
                                           min_value=0, max_value=25, step=1, value=num_delayed_payments_default)
    changed_credit_limit = st.number_input('Please enter the percentage change in credit card limit.',
                                           min_value=0.0, max_value=35.000, step=0.01, value=changed_credit_limit_default)

    num_credit_inqueries = st.number_input('Please enter the number of credit card inquiries.',
                                           min_value=0, max_value=20, step=1, value=num_credit_inqueries_default)
    credit_mix = st.selectbox('Please select your classification of the mix of credits.',
                              ('Standard', 'Bad', 'Good'))
    outstanding_debt = st.number_input('Please enter the remaining debt (Outstanding debt) to be paid.',
                                       min_value=0.00, max_value=6000.0, step=.01, value=outstanding_debt_default)

    credit_history_age = st.number_input('Please enter how many months old is your credit history (Credit history age)',
                                         min_value=0, max_value=500, step=1, value=credit_history_age_default)

    payment_min_amount = st.selectbox('Please select whether the minimum amount was paid.', [
        'Yes', 'No', 'NM'])
    total_emi_per_month = st.number_input('please enter the total_emi_per_month.',
                                          min_value=0.00, max_value=5000.00, step=.01, value=total_emi_per_month_default)
    amount_invested_monthly = st.number_input('Please enter the monthly amount invested by you.',
                                              min_value=0.00, max_value=450.00, step=.01, value=amount_invested_monthly_default)

    run = st.button('Predict Credit Score')

st.header('Credit Score Results')

col1, col2 = st.columns([3, 2])

with col2:
    x1 = [0, 6, 0]
    x2 = [0, 4, 0]
    x3 = [0, 2, 0]
    y = ['0', '1', '2']

    f, ax = plt.subplots(figsize=(5, 2))

    p1 = sns.barplot(x=x1, y=y, color='#3EC300')
    p1.set(xticklabels=[], yticklabels=[])
    p1.tick_params(bottom=False, left=False)
    p2 = sns.barplot(x=x2, y=y, color='#FAA300')
    p2.set(xticklabels=[], yticklabels=[])
    p2.tick_params(bottom=False, left=False)
    p3 = sns.barplot(x=x3, y=y, color='#FF331F')
    p3.set(xticklabels=[], yticklabels=[])
    p3.tick_params(bottom=False, left=False)

    plt.text(0.7, 1.05, "POOR", horizontalalignment='left',
             size='medium', color='white', weight='semibold')
    plt.text(2.3, 1.05, "STANDARD", horizontalalignment='left',
             size='medium', color='white', weight='semibold')
    plt.text(4.7, 1.05, "GOOD", horizontalalignment='left',
             size='medium', color='white', weight='semibold')

    ax.set(xlim=(0, 6))
    sns.despine(left=True, bottom=True)

    figure = st.pyplot(f)

with col1:

    placeholder = st.empty()

    if run:
        response = {
            'annual_income': annual_income,
            'monthly_inhand_salary': monthly_inhand_salary,
            'num_accounts': num_accounts,
            'num_credit_cards': num_credit_cards,
            'interest_rate': interest_rate,
            'num_loans': num_loans,
            'delay_from_due_date': delay_from_due_date,
            'num_delayed_payments': num_delayed_payments,
            'changed_credit_limit': changed_credit_limit,
            'num_credit_inqueries': num_credit_inqueries,
            'credit_mix': credit_mix,
            'outstanding_debt': outstanding_debt,
            'credit_history_age': credit_history_age,
            'payment_min_amount': payment_min_amount,
            'total_emi_per_month': total_emi_per_month,
            'amount_invested_monthly': amount_invested_monthly
        }
        input_df = pd.DataFrame({
            'feature': response.keys(),
            'value': response.values(),
        })
        output = transform_response(response)
        output = np.array(output).reshape((1, config.num_features))

        pred = model.predict(output)[0]
        credit_score = config.output_transform_dict[pred]

        if credit_score == 'Good':
            st.balloons()
            t1 = plt.Polygon([[5, 0.5], [5.5, 0], [4.5, 0]], color='black')
            st.code(
                'Your credit score is GOOD! Congratulations!')
            st.code('''The risk of extending credit to this person  
is minimal since their credit score suggests 
that they are likely to repay a loan.''')
        elif credit_score == 'Standard':
            t1 = plt.Polygon([[3, 0.5], [3.5, 0], [2.5, 0]], color='black')
            st.code('Your credit score is Standard', language="python")
            st.code('''
                    This individual has a good chance of repaying a loan,
although they occasionally fall behind on payments, 
according to their credit score. Giving them 
credit carries a medium level of risk.
                    ''', language='python')
        elif credit_score == 'Poor':
            t1 = plt.Polygon([[1, 0.5], [1.5, 0], [0.5, 0]], color='black')
            st.code('Your credit score is POOR.', language="python")
            st.code('''
The danger of extending credit to this person is
considerable since their credit score suggests 
that they are unlikely to repay a loan.''')
        plt.gca().add_patch(t1)
        figure.pyplot(f)
        prob_fig, ax = plt.subplots()

        with st.expander('Click to see your given input features'):
            st.dataframe(input_df)

        with st.expander('Click to see how certain the algorithm was'):
            plt.pie(model.predict_proba(output)[0], labels=[
                    'Good', 'Poor', 'Standard'], autopct='%.0f%%')
            st.pyplot(prob_fig)

        with st.expander('Click to see feature importances'):
            importance = model.feature_importances_
            # importance = np.exp(importance)
            imp_df = pd.DataFrame({
                'feature': cols_order,
                'importance': importance,
            })
            imp_df.sort_values(by='importance', ascending=False, inplace=True)

            importance_figure, ax = plt.subplots()
            bars = ax.barh('feature', 'importance', data=imp_df)
            ax.bar_label(bars)
            plt.ylabel('importance', fontweight='bold')
            plt.xlabel('feature', fontweight='bold')
            plt.yticks(imp_df['feature'], weight='bold')
            sns.despine(right=True, top=True)
            st.pyplot(importance_figure)
