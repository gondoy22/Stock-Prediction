# Stock-Prediction
# Stock Prediction Project

## Overview

This project is designed to evaluate the future values of stock prices using time series forecasting. The model is built using data from Yahoo Finance and FRED APIs. The main objective is to build an LSTM (Long Short-Term Memory) model to predict future stock prices based on historical data and other relevant economic indicators.

### **Data Sources**
The data used in this project is sourced from:
- **Yahoo Finance API** (for stock prices)
- **FRED API** (for economic indicators)

### **Important Notes**
- The project requires an **API Key** from FRED API. You will need to create your own key by signing up on the [FRED website](https://fred.stlouisfed.org/) to access the data. (at least i think so, there is a slight chance it will work without it. Just give it a try:))
- The project uses a **model-building function** which, once executed, generates a trained model. This model is saved as a `.pkl` file in the `models` folder, which will be created automatically. Due to the model size, it is not included in the repository.
- **Model Saving and Loading**: After building the model, the functions will automatically load the saved model, eliminating the need to re-build it every time. This is important because model training can be time-consuming.

### **Project Structure**

The project is organized into the following sections:

1. **Data Cleaning and EDA**:
    - The dataset is merged, cleaned, imputed, and prepared for model building.
    - Data preparation involves handling missing values, outliers, and irrelevant variables.

2. **Exploratory Data Analysis (EDA)**:
    - This section visualizes the data and helps understand the structure and relationships within the dataset.
    - Based on the EDA, certain variables are deleted to optimize the modeling process.

3. **Functions**:
    - This folder contains all the functions used throughout the project.
    - Functions are modular and can be reused for different data sets or experiments.

4. **Notebook**:
    - The Jupyter notebook in this folder provides an overview of the entire project.
    - It includes the usage of various functions and demonstrates how the model works, with a focus on predicting stock prices for three companies. 
    - **Note**: The last section of the notebook is still a work in progress.

### **How to Use This Project**

1. **Install Required Libraries**:
    - Make sure you have all the necessary libraries installed. You can find the required libraries in the `requirements.txt` file.

2. **Obtain an API Key**:
    - To access the FRED API data, you will need to get your own API key from the [FRED API website](https://fred.stlouisfed.org/).

3. **Run the Notebook**:
    - Execute the Jupyter notebook to see the workflow. The notebook demonstrates the steps from data cleaning to model building and evaluation.
    
4. **Model Building**:
    - When you first run the project, the model will be built using the provided `build_model()` function. This function will save the trained model as a `.pkl` file inside the `models` folder.
    - After the model is saved, the rest of the functions will simply load the model, which significantly reduces execution time for future runs.

---

### **Acknowledgments**
- **Yahoo Finance API** and **FRED API** were instrumental in gathering the stock and economic data used in this project.
- Thanks to the open-source community for providing the tools and libraries that made this project possible.
