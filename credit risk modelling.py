import numpy as np
import pandas as pd
from scipy.stats import norm

# Assume df has been loaded and renamed correctly
df = pd.read_csv("american_bankruptcy.csv")
columns = {
    'X1': 'Current_Assets',
    'X2': 'Cost_of_Goods_Sold',
    'X3': 'Depreciation_And_Amortization',
    'X4': 'EBITDA',
    'X5': 'Inventory',
    'X6': 'Net_Income',
    'X7': 'Total_Receivables',
    'X8': 'Market_Value',
    'X9': 'Net_Sales',
    'X10': 'Total_Assets',
    'X11': 'Total_Long_Term_Debt',
    'X12': 'EBIT',
    'X13': 'Gross_Profit',
    'X14': 'Total_Current_Liabilities',
    'X15': 'Retained_Earnings',
    'X16': 'Total_Revenue',
    'X17': 'Total_Liabilities',
    'X18': 'Total_Operating_Expenses'
}
df = df.rename(columns=columns)


# Function to calculate Merton Model
def calculate_merton_model(dataframe):
    results = []
    r = 0.05  # Risk-free rate
    dataframe = dataframe.copy()
    dataframe['Returns'] = dataframe['Total_Assets'].pct_change().dropna()
    volatility = dataframe['Returns'].std()
    T = 1  # Time to maturity (1 year)

    for year in dataframe['year'].unique():
        data = dataframe[dataframe['year'] == year]
        V = data['Total_Assets'].values[0]
        D = data['Total_Current_Liabilities'].values[0]

        d1 = (np.log(V / D) + (r + 0.5 * volatility ** 2) * T) / (volatility * np.sqrt(T))
        d2 = d1 - volatility * np.sqrt(T)

        equity_value = V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2)
        pd_value = norm.cdf(-d2)

        results.append({
            'year': int(year),  # Make sure it's an int, not np.int64
            'equity_value': float(equity_value),  # Convert to standard float
            'default_probability': float(pd_value)  # Convert to standard float
        })

    # Convert list of dicts to DataFrame
    return pd.DataFrame(results)

list_of_defaulted_companies = [
    company for company in df['company_name'].unique()
    if 'failed' in df[df['company_name'] == company]['status_label'].unique()
]
print(list_of_defaulted_companies)


# Applying the function to a specific company
C_300 = df[df['company_name'] == 'C_300']
C_300_merton = calculate_merton_model(C_300)

# Output the resulting DataFrame
print(C_300_merton)







