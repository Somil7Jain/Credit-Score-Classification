cols_to_be_scaled = ['annual_income', 'monthly_inhand_salary', 'outstanding_debt', 'credit_history_age',
                     'total_emi_per_month', 'amount_invested_monthly']

cols_order = ['annual_income',	'monthly_inhand_salary', 'num_accounts', 'num_credit_cards', 'interest_rate',	'num_loans',	'delay_from_due_date',	'num_delayed_payments',
              'changed_credit_limit',	'num_credit_inqueries',	'credit_mix',	'outstanding_debt',	'credit_history_age',	'payment_min_amount',	'total_emi_per_month',	'amount_invested_monthly']

mean_dict = {
    'annual_income': 50505.123449,
    'monthly_inhand_salary': 4197.270835,
    'outstanding_debt': 1426.220376,
    'credit_history_age': 221.220460,
    'total_emi_per_month': 107.699208,
    'amount_invested_monthly': 55.101315
}
std_dict = {
    'annual_income':  38299.422093,
    'monthly_inhand_salary':  3186.432497,
    'outstanding_debt':  1155.129026,
    'credit_history_age': 99.680716,
    'total_emi_per_month': 132.267056,
    'amount_invested_monthly':  39.006932
}
transform_dict = {
    'credit_mix': {
        'Bad': 0,
        'Good': 1,
        'Standard': 2
    },
    'payment_min_amount': {
        'NM': 0,
        'No': 1,
        'Yes': 2
    },
    'credit_score': {
        'Good': 0,
        'Poor': 1,
        'Standard': 2
    }
}


def transform_response(response):
    output = list()
    for col in cols_order:
        if col in list(transform_dict.keys()):
            output.append(transform_dict[col][response[col]])
        elif col in cols_to_be_scaled:
            output.append((response[col]-mean_dict[col])/std_dict[col])
        else:
            output.append(response[col])
    return output
