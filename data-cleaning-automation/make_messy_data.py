"""
Generates a realistic 'messy' raw version of Car_Sales_Dataset.csv, simulating
common real-world data entry problems: missing values, duplicate rows,
inconsistent text casing/whitespace, inconsistent date formats, and invalid
numeric entries. This messy file plays the role of 'raw historical data'
that the automation pipeline (data_cleaning_automation.py) will clean.
"""

import pandas as pd
import numpy as np

np.random.seed(7)

df = pd.read_csv("Car_Sales_Dataset.csv")
messy = df.copy()

# 1. Inject missing values in a few columns (realistic sparsity)
for col, frac in [('Unit Price (INR)', 0.05), ('Discount (%)', 0.06),
                   ('Fuel Type', 0.04), ('Region', 0.03)]:
    idx = messy.sample(frac=frac, random_state=hash(col) % 1000).index
    messy.loc[idx, col] = np.nan

# 2. Inject duplicate rows (exact + near-duplicate with different casing)
dupes = messy.sample(6, random_state=1).copy()
messy = pd.concat([messy, dupes], ignore_index=True)

near_dupes = messy.sample(4, random_state=2).copy()
near_dupes['Fuel Type'] = near_dupes['Fuel Type'].astype(str).str.upper()
messy = pd.concat([messy, near_dupes], ignore_index=True)

# 3. Inconsistent text casing / whitespace in categorical columns
def messify_text(val, mode):
    if pd.isna(val):
        return val
    if mode == 'upper':
        return str(val).upper()
    if mode == 'lower':
        return str(val).lower()
    if mode == 'space':
        return f"  {val}  "
    return val

n = len(messy)
rng = np.random.default_rng(42)
modes = rng.choice(['upper', 'lower', 'space', 'none'], size=n, p=[0.12, 0.12, 0.1, 0.66])
messy['Fuel Type'] = [messify_text(v, m) for v, m in zip(messy['Fuel Type'], modes)]

modes2 = rng.choice(['upper', 'lower', 'space', 'none'], size=n, p=[0.1, 0.1, 0.1, 0.7])
messy['Region'] = [messify_text(v, m) for v, m in zip(messy['Region'], modes2)]

modes3 = rng.choice(['space', 'none'], size=n, p=[0.15, 0.85])
messy['Dealer'] = [messify_text(v, m) for v, m in zip(messy['Dealer'], modes3)]

# 4. Inconsistent date formats (mix of ISO, DD-MM-YYYY, DD/MM/YYYY)
def messify_date(d, fmt_choice):
    d = pd.to_datetime(d)
    if fmt_choice == 0:
        return d.strftime('%Y-%m-%d')
    elif fmt_choice == 1:
        return d.strftime('%d-%m-%Y')
    else:
        return d.strftime('%d/%m/%Y')

fmt_choices = rng.choice([0, 1, 2], size=n, p=[0.6, 0.2, 0.2])
messy['Date'] = [messify_date(d, c) for d, c in zip(messy['Date'], fmt_choices)]

# 5. Invalid / out-of-range numeric entries (data entry errors)
err_idx = messy.sample(3, random_state=3).index
messy.loc[err_idx, 'Discount (%)'] = -5  # invalid negative discount

err_idx2 = messy.sample(2, random_state=4).index
messy.loc[err_idx2, 'Units Sold'] = 0  # invalid zero units on a recorded sale

# Shuffle rows so duplicates aren't neatly at the end (realistic raw export)
messy = messy.sample(frac=1, random_state=5).reset_index(drop=True)

messy.to_csv("Car_Sales_RAW_messy.csv", index=False)
print(f"Messy raw dataset created: {messy.shape[0]} rows (from {df.shape[0]} clean rows)")
print("\nMissing values introduced:\n", messy.isnull().sum())
print("\nExact duplicate rows:", messy.duplicated().sum())
