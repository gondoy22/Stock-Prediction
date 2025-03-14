import pandas as pd
import numpy as np
import joblib
import os


def results(companies,date_start):
    loaded_models = {}
    for name in companies:
        model_path = os.path.join("models", f"{name}_model.pkl")
        loaded_models[name] = joblib.load(model_path)

    df_results = pd.DataFrame()

    for company in companies:

        pred = predictions([company], date_start)
        df_results[company] = pred[company]
        df_results.index = test_dates

    return df_results