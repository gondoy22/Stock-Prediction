import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def predict_future_x_days(model,company_name, date_start, x_days):
"""
Function that returns a prediction of actual future for x_days forward
Args:
    model: a model based on which we want to get our predictions
    company_name: name of the company for which we want to retrieve data. e.g. 'TSLA'
    date_start: date from which we want to retrieve data e.g '2019-01-01'
    x_days: number of days in future we want to forecast stock prices

Example of usage: predict_future_x_days(TSLA_model.pkl,'TSLA', '2019-01-01', 20)

"""
    _, _,_,_,_,scaler_y,scaler_merged, df_merged = datasets(company_name, date_start)
    df_all_no_date = df_merged.drop(columns='Date')
    scaled_all = pd.DataFrame(scaler_merged.transform(df_all_no_date), columns=df_all_no_date.columns)
    X_all = scaled_all.to_numpy() 

    #Preparing last 50 days to predict next day, because our model uses past 50 days

    last_50_days = X_all[-50:].reshape((1, 50, X_all.shape[-1])) 
    predictions = []
    current_sequence = last_50_days.copy()

    for _ in range(x_days):

        next_pred = model.predict(current_sequence, verbose=0)
        predictions.append(next_pred[0, 0])
        #Deleting first row of current sequence so we can squeeze in one new row at the front
        next_input = current_sequence[0, 1:, :] 
        #Adding new row at the front so our models still gets 50 elements.
        new_row = current_sequence[0, -1, :].copy() 
        new_row[0] = next_pred[0, 0]  
        current_sequence = np.append(next_input, [new_row], axis=0).reshape((1, 50, X_all.shape[-1]))

    predictions_unscaled = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)).ravel()
    last_date = pd.to_datetime(df_merged['Date'].iloc[-1])
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=x_days, freq='B')

    return pd.DataFrame({'Date': future_dates, 'Predicted_Close': predictions_unscaled})



def results_future_x_days_all_companies(companies, date_start, x_days):
 """
Function that returns a prediction of actual future for x_days forward for multiple companies
Args:
    companies: List of companies for which we want to build a model. e.g. ['TSLA','NVDA']
    date_start: date from which we want to retrieve data e.g '2019-01-01'
    x_days: number of days in future we want to forecast stock prices

Example of usage: predict_future_x_days_all_companies(['TSLA','NVDA'], '2019-01-01', 20)

"""
    df_future_results = pd.DataFrame()
    loaded_models = {}
    for name in companies:
        model_path = os.path.join("models", f"{name}_model.pkl")
        loaded_models[name] = joblib.load(model_path)

    #Getting Date column from one of the models that so we can index our data by it.
    first_company_predictions = predict_future_x_days(loaded_models[companies[0]],companies[0] ,date_start, x_days)
    df_future_results['Date'] = first_company_predictions['Date']

    #Getting prediction for all of the companies listed
    for company in companies:
        predictions = predict_future_x_days(loaded_models[company],company, date_start, x_days)
        df_future_results[company] = predictions['Predicted_Close']

    df_future_results.set_index('Date', inplace=True)

    return df_future_results
