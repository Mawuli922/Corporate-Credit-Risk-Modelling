import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("corporate_rating.csv")

print(df.info())

print(df['Rating'].unique())


def convert_to_numeric(x):
    if x == 'AAA':
        return 0
    elif x == 'AA':
        return 1
    elif x == 'A':
        return 2
    elif x == 'BBB':
        return 3
    elif x == 'BB':
        return 4
    elif x == 'B':
        return 5
    elif x == 'CCC':
        return 6
    elif x == 'CC':
        return 7
    elif x == 'C':
        return 8
    else:
        return 9


df['credit_rating_numeric'] = df['Rating'].apply(lambda x: convert_to_numeric(x))

print(df['Rating'].value_counts())
df['credit_rating_numeric'].value_counts().sort_index().plot(kind='bar')
plt.show()
