import pandas as pd
import numpy as np
import yfinance as yf
import fredapi
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import os
temp1_cache = {}
temp_cache = {}


def datasets(company_name, date_start):

"""
Function that returns crucial data sets for model building phase or visualization of model's prediction

Args:
    company_name: Name of the company for which we want to retrieve data. e.g. 'TSLA'
    date_start: date from which we want to retrieve data e.g '2019-01-01'

Example of usage: datasets('TSLA','2019-01-01')

"""




    #Loading datasets from FRED and yahoo api
    load_dotenv()
    cache_key = (company_name, date_start)
    if cache_key in temp1_cache:
        return temp1_cache[cache_key]

    df_company = yf.download(company_name, start=date_start)
    df_company = df_company.reset_index().rename(columns={"index": "Date"})

    API_KEY = os.getenv('API_KEY')
    fred = fredapi.Fred(api_key=API_KEY)
    gdp_data = fred.get_series("GDP")
    cpi_data = fred.get_series("CPIAUCSL")
    unemployment_data = fred.get_series("UNRATE")
    fed_funds_data = fred.get_series("FEDFUNDS")
    sp500_data = fred.get_series("SP500")
    vix_data = fred.get_series("VIXCLS")

    df_economic = pd.DataFrame({
        "SP500": sp500_data,
        "VIX": vix_data,
        "GDP": gdp_data,
        "CPI": cpi_data,
        "UR": unemployment_data,
        "FF": fed_funds_data})
    
    #Data cleaning performed according to the rules stated in Data_cleaning 

    df_economic = df_economic.reset_index().rename(columns={"index": "Date"})
    df_economic = df_economic[df_economic['Date'] > date_start]
    df_company_temp = df_company.copy()
    df_company_temp.columns = df_company.columns.droplevel(1)
    df_company = df_company_temp
    df_merged = pd.merge(df_company, df_economic, on='Date', how='left')
    numeric_cols = df_merged.columns.drop('Date')
    df_merged[numeric_cols] = df_merged[numeric_cols].astype('float64')
     numeric_data = df_merged.drop(columns=['Date'])
    imputer = KNNImputer(n_neighbors=2)
    df_merged_imputed = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_data.columns)
    df_merged.iloc[:, 1:] = df_merged_imputed

    #According to EDA we delete highly correlated variables and prepare train and test sets for LSTM model.

    df_merged = df_merged.drop(columns=['High', 'Low', 'Open'])
    test_percent = 0.2
    test_point = np.round(len(df_merged) * test_percent)
    test_index = int(len(df_merged) - test_point)

    train = df_merged.iloc[:test_index]
    test = df_merged.iloc[test_index:]

    train_no_date = train.drop(columns='Date')
    test_no_date = test.drop(columns='Date')

    _, y_train = train_no_date.drop(columns='Close'), train_no_date['Close']
    _, y_test = test_no_date.drop(columns='Close'), test_no_date['Close']

    #Definig scalers

    scaler_y = MinMaxScaler()
    scaler_merged = MinMaxScaler()

    scaled_merged_train = pd.DataFrame(scaler_merged.fit_transform(train_no_date), columns=train_no_date.columns, index=train_no_date.index)
    scaled_merged_test = pd.DataFrame(scaler_merged.transform(test_no_date), columns=test_no_date.columns, index=test_no_date.index)

    scaled_y_train = pd.Series(scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten(), index=y_train.index, name='Close')
    scaled_y_test = pd.Series(scaler_y.transform(y_test.values.reshape(-1, 1)).flatten(), index=y_test.index, name='Close')
    X_train_array = scaled_merged_train.to_numpy()

    scaled_y_test = scaled_y_test.reset_index(drop=True)
    X_test_array = scaled_merged_test.to_numpy()

    temp1_cache[cache_key] = (X_train_array, scaled_y_train, X_test_array,
                              scaled_y_test, scaled_merged_train, scaler_y, scaler_merged, df_merged)
    return temp1_cache[cache_key]