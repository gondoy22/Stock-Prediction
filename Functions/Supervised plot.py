import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

def plot_stock(companies, date_start):
    fig, axes = plt.subplots(nrows=len(companies), ncols=1, figsize=(20, 12))
    prediction = predictions(companies, date_start)

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for i, company in enumerate(companies):
        ax = axes[i]

        _, _, _, _, _, _, _, df = datasets(company, date_start)

        # Get the length of predictions for this company
        pred_len = len(prediction[company])

        # Adjust test_dates to match the length of predictions
        test_dates = df['Date'][-pred_len:]

        ax.plot(df['Date'], df['Close'], color='blue')
        # Use adjusted test_dates for plotting predictions
        ax.plot(test_dates, prediction[company], color='red')
        ax.set_title(f'Stock Price Prediction for {company}')
        ax.set_ylabel('Price', fontsize=14)
        if i == 0:
            ax.legend(['Actual', 'Predicted'])


    axes[-1].set_xlabel('Date', fontsize=14)

    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    plt.show()