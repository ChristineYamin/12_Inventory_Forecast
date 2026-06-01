import pandas as pd
import numpy as np

# 1. Setup Parameters
num_rows = 2000
items = ['Ring', 'Earring', 'Necklace', 'Bracelet', 'Gold Bar', 
         'Pendant', 'Anklet', 'Bangle', 'Brooch', 'Charm']

# 2. Generate Base Data
np.random.seed(42)  # Ensures you get the same 'random' data every time
data = {
    'date': pd.date_range(start='2025-01-01', periods=num_rows, freq='D'),
    'item_name': np.random.choice(items, num_rows),
    'category': 'Gold Jewelry',
    'quantity_sold': np.random.randint(0, 15, num_rows),
    'unit_price': np.random.randint(2000000, 5000000, num_rows)
}
df = pd.DataFrame(data)

# 3. Add Professional "Real-World" Complexities
# A. Add Seasonal Spike (Wedding/Festival season: Oct, Nov, Dec)
df.loc[df['date'].dt.month.isin([10, 11, 12]), 'quantity_sold'] *= 2

# B. Add Price Drift (Market price trend)
df['unit_price'] = df['unit_price'] + (df.index * 500)

# C. Add "Closed Shop" days (No sales)
closed_days = df.sample(frac=0.05).index
df.loc[closed_days, 'quantity_sold'] = 0

# D. Add Missing Values (Human entry gaps)
missing_idx = df.sample(frac=0.01).index
df.loc[missing_idx, 'unit_price'] = np.nan

# E. Add Outliers (Typo errors)
df.loc[0, 'unit_price'] = 999999999 

# F. Add Stock Logic (Stock = Previous stock - sold + restock)
# This is a simple simulation for the 'stock_left' column
df['stock_left'] = 100 - df.groupby('item_name')['quantity_sold'].cumsum() % 100

# 4. Save and Verify
df.to_csv('gold_stock_data.csv', index=False)

print(f"Dataset generated successfully with {len(df)} rows.")
print("Check gold_stock_data.csv in the folder.")