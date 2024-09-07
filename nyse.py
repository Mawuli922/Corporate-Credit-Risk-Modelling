import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

df = pd.read_csv("fundamentals.csv", parse_dates=["Period Ending"])
print(df.info())

print(df['Ticker Symbol'].value_counts())

df['year'] = df["Period Ending"].dt.year
print(df['year'].value_counts())

def calculate_merton_model(df):
    results = []
    r = 0.05
    df = df.copy()
    df['Returns'] = df['Total Assets'].pct_change().dropna()
    volatility = df['Returns'].std()
    T=1

    for year in df['year'].unique():
        data = df[df['year'] == year]
        V = data['Total Assets'].values[0]
        D = data['Total Current Liabilities'].values[0]
        tc= data['Ticker Symbol'].values[0]
        d1 = (np.log(V / D) + (r*0.5*volatility**2)*T)/ (volatility*np.sqrt(T))
        d2 = d1 - volatility*np.sqrt(T)

        equity_value = V*norm.cdf(d1) - D*np.exp(-r*T) * norm.cdf(d2)
        pd = norm.cdf(-d2)
        results.append({
            'year': int(year),
            'equity_value': float(equity_value),
            'merton_default_probability': float(pd),
            'ticker_symbol': tc
        })

    return results

def convert_to_dataframe(x):
    return pd.DataFrame(x)
dataframe_list = []
for company in df['Ticker Symbol'].unique():
    company_df = df[df['Ticker Symbol']== company]
    dataframe_list.append(convert_to_dataframe(calculate_merton_model(company_df)))
new_df = pd.concat(dataframe_list).reset_index()
print(new_df['merton_default_probability'].describe())

print(new_df.head())

new_df['merton_default_probability'].hist(bins=100)
plt.show()



print(new_df.sort_values(by="merton_default_probability", ascending=False).head(20))


