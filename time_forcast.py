import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# -----------------------
# Step 1: Simulate Sales Data
# -----------------------
date_rng = pd.date_range(start='2018-01-01', end='2022-12-31', freq='M')
np.random.seed(42)
sales_data = np.random.normal(loc=20000, scale=3000, size=len(date_rng)).astype(int)

df = pd.DataFrame({'Date': date_rng, 'Sales': sales_data})
df.set_index('Date', inplace=True)

# -----------------------
# Step 2: Fit ARIMA Model
# -----------------------
# ARIMA(p,d,q) — (2,1,2) is a reasonable starting point
model_arima = ARIMA(df['Sales'], order=(2,1,2))
arima_result = model_arima.fit()

# -----------------------
# Step 3: Forecast Future Sales
# -----------------------
forecast_steps = 12
forecast = arima_result.forecast(steps=forecast_steps)

# -----------------------
# Step 4: Plot Historical & Forecasted Sales
# -----------------------
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Sales'], label='Historical Sales')

# Create future date range for the forecast
forecast_index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='M')
plt.plot(forecast_index, forecast, label='ARIMA Forecast', color='red')

plt.title('ARIMA Sales Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

