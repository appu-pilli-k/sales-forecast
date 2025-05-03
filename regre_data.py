import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Simulate monthly sales data
date_rng = pd.date_range(start='2018-01-01', end='2022-12-31', freq='M')
np.random.seed(42)
sales_data = np.random.normal(loc=20000, scale=3000, size=len(date_rng)).astype(int)

# Create DataFrame
df = pd.DataFrame({'Date': date_rng, 'Sales': sales_data})
df.set_index('Date', inplace=True)

# -------------------------------
# Regression Modeling
# -------------------------------
# Feature Engineering
df_ml = df.copy()
df_ml['Month'] = df_ml.index.month
df_ml['Year'] = df_ml.index.year
df_ml['Time'] = np.arange(len(df_ml))  # numeric time for regression

X = df_ml[['Month', 'Year', 'Time']]
y = df_ml['Sales']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"Linear Regression MSE: {mse:.2f}")

# Plot Predictions
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, y_pred, label="Predicted", linestyle='--')
plt.title("Linear Regression Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

