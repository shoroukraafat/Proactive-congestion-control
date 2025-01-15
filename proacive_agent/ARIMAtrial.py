import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
import matplotlib.pyplot as plt
import datetime
import time
import pickle
import re
import joblib
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard,ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

warnings.filterwarnings('ignore')

# define a function to convert a vector of time series into a 2D matrix
def convertSeriesToMatrix(vectorSeries, sequence_length):
    matrix=[]
    for i in range(len(vectorSeries)-sequence_length+1):
        matrix.append(vectorSeries[i:i+sequence_length])
    return matrix

# random seed
np.random.seed(1234)

# load the data
# Read the data from an Excel sheet
#df =  pd.read_csv('Rewards_t_1726332254.csv', sep=',', low_memory=False, encoding='utf-8', on_bad_lines='skip')

df =  pd.read_csv('myTableNew.csv', sep=',', low_memory=False, encoding='utf-8', on_bad_lines='skip')

# using the first row of the sheet that represents the downlink throughput
# Convert column names to float32 with proper approximation
column_names = []
for column_name in df.columns:
    try:
        numeric_value = re.findall(r'\d+\.\d+', str(column_name))[0]
        float_value = np.float32(numeric_value)
        column_names.append(float_value)
    except (ValueError, IndexError):
        pass
# Print the converted column names
data = np.array(column_names)

print(len(data))


def arima_forecast(data, forecast_steps=5):
    """
    Fits an ARIMA model to the input data and forecasts future values.
    
    Parameters:
    data (list or array): Time series data (55 numbers)
    forecast_steps (int): Number of steps to forecast (default: 5)
    
    Returns:
    dict: Contains forecasted values, confidence intervals, and model metrics
    """
    # Convert input to pandas Series
    series = pd.Series(data)
    
    # Grid search for best parameters
    best_aic = float('inf')
    best_params = None
    
    # Try different combinations of p, d, q
    for p in range(0, 3):
        for d in range(0, 2):
            for q in range(0, 3):
                try:
                    model = ARIMA(series, order=(p, d, q))
                    results = model.fit()
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_params = (p, d, q)
                except:
                    continue
    
    # Fit the model with best parameters
    final_model = ARIMA(series, order=best_params)
    final_results = final_model.fit()
    joblib.dump(final_results, "arima_resultsupdated.pkl")
    
    # Make predictions
    forecast = final_results.forecast(steps=forecast_steps)
    forecast_ci = final_results.get_forecast(steps=forecast_steps).conf_int()
    
    # Calculate model metrics
    fitted_values = final_results.fittedvalues
    mse = mean_squared_error(series[1:], fitted_values[:-1])
    rmse = np.sqrt(mse)
    
    with open('best_arima_model.pkl', 'wb') as file:
            pickle.dump(model, file)
    
    return {
        'forecast': forecast.values,
        'confidence_intervals': {
            'lower': forecast_ci.iloc[:, 0].values,
            'upper': forecast_ci.iloc[:, 1].values
        },
        'model_params': {
            'order': best_params,
            'aic': best_aic,
            'rmse': rmse
        }
    }

# Example usage
if __name__ == "__main__":
    # Example data (replace with your 55 numbers)
    
    # Get forecasts
    results = arima_forecast(data)
    
    # Print results
    print("\nForecasted values:")
    for i, val in enumerate(results['forecast'], 1):
        print(f"Step {i}: {val:.2f}")
    
    print("\nConfidence Intervals:")
    for i in range(len(results['forecast'])):
        print(f"Step {i+1}: ({results['confidence_intervals']['lower'][i]:.2f}, "
              f"{results['confidence_intervals']['upper'][i]:.2f})")
    
    print("\nModel Information:")
    print(f"Best ARIMA Order: {results['model_params']['order']}")
    print(f"AIC: {results['model_params']['aic']:.2f}")
    print(f"RMSE: {results['model_params']['rmse']:.2f}")
