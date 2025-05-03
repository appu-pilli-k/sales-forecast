import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Simulate monthly sales data
date_rng = pd.date_range(start='2018-01-01', end='2022-12-31', freq='M')
np.random.seed(42)
sales_data = np.random.normal(loc=20000, scale=3000, size=len(date_rng)).astype(int)

# Create DataFrame
df = pd.DataFrame({'Date': date_rng, 'Sales': sales_data})
df.set_index('Date', inplace=True)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Sales'], label='Monthly Sales')
plt.title('Monthly Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

