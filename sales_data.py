import pandas as pd
import numpy as np

# Simulate monthly sales data for 5 years
date_rng = pd.date_range(start='2018-01-01', end='2022-12-31', freq='M')
np.random.seed(42)
sales_data = np.random.normal(loc=20000, scale=3000, size=len(date_rng)).astype(int)

df = pd.DataFrame({'Date': date_rng, 'Sales': sales_data})
df.set_index('Date', inplace=True)

print(df.head())

