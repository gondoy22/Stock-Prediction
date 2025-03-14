import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os


def plot_stock_with_future(companies, date_start, x_days):

    fig, axes = plt.subplots(nrows=len(companies), ncols=1, figsize=(20, 12))
    future_predictions = results_future_x_days_all_companies(companies, date_start, x_days)

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for i, company in enumerate(companies):
        ax = axes[i]

        _, _, X_test_array, scaled_y_test, _, scaler_y, _, df = datasets(company, date_start)

        # Extract test dates from the data frame
        test_dates = df['Date'].iloc[-len(X_test_array):] 

        #Filtering data for plotting
        df_test = df[df['Date'].isin(test_dates)]
        future_predictions_test = future_predictions[future_predictions.index.isin(future_predictions.index)]  # Ensures dates align

        # Rescale y_test
        y_test_original = scaler_y.inverse_transform(scaled_y_test.values.reshape(-1, 1)).ravel()

        #Plotting the data we now and first x days of our unsupervised predictions
        ax.plot(df_test['Date'], df_test['Close'], color='blue', label='Actual')
        ax.plot(future_predictions_test.index, future_predictions_test[company], color='green', label=f'Future Predictions ({x_days} days)')

        ax.set_title(f'Stock Price Prediction for {company} (Test Data)')
        ax.set_ylabel('Price', fontsize=14)
        if i == 0:
            ax.legend()

    axes[-1].set_xlabel('Date', fontsize=14)
    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    plt.show()
