import pandas as pd
import numpy as np
import joblib
import os


def results(companies,date_start):
"""
Function that returns a data frame with predicted stock values on test set
Args:
    companies: List of companies for which we want to build a model. e.g. ['TSLA','NVDA']
    date_start: date from which we want to retrieve data e.g '2019-01-01'

Example of usage: results(['TSLA','NVDA'], '2019-01-01')

"""
    loaded_models = {}
    for name in companies:
        model_path = os.path.join("models", f"{name}_model.pkl")
        loaded_models[name] = joblib.load(model_path)

    df_results = pd.DataFrame()

    for company in companies:

        pred = predictions([company], date_start)
        df_results[company] = pred[company]
        test_dates = df_merged['Date'].iloc[-len(X_test_array)+length:]
        df_results.index = test_dates

    return df_results