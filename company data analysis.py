import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import silhouette_score



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

print(df.head())
print(df.info())
plt.figure(figsize=(10, 6))
df['year'].value_counts().plot(kind='bar')
plt.ylabel("Company Count")
plt.title("Count of Companies per year")
plt.show()


corr_mat = df.drop(["company_name", "status_label", "year"], axis=1).corr().abs()
plt.figure(figsize=(15, 15))
sns.heatmap(corr_mat, cmap="Blues", annot=True, fmt=".2f")
plt.title("Pearson's Correlation Matrix of Financial Metrics")
plt.show()

upper = corr_mat.where(np.triu(np.ones(corr_mat.shape),k=1).astype(bool))

to_drop = [column for column in upper.columns if any(upper[column]>0.9)]
reduced_df = df.drop(df[to_drop], axis=1)

print(reduced_df.info())

features_to_scale = reduced_df.drop(['company_name', 'status_label', 'year'], axis=1)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_to_scale)

selector = VarianceThreshold(threshold=0.1)
selected_features = selector.fit_transform(scaled_features)
print(scaled_features.shape)
print(selected_features.shape)
year_k_best = {}
for year in reduced_df['year'].unique():
    df_filtered = reduced_df[reduced_df['year'] == year]
    indices = df_filtered.index
    filtered_selected_features = selected_features[indices, :]
    silhouette_scores = []
    inertia_scores = {}
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(filtered_selected_features)
        silhouette_avg = silhouette_score(filtered_selected_features, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        inertia_scores[k] = kmeans.inertia_

    plt.bar(inertia_scores.keys(), inertia_scores.values())
    plt.ylabel("inertia score")
    plt.xlabel("number of clusters")
    plt.title(f"Inertia score per cluster for the year {year}")
    plt.show()



    for index, num in enumerate(silhouette_scores):
        if num == max(silhouette_scores):
            year_k_best[year] = index+2


plt.bar(year_k_best.keys(), year_k_best.values())
plt.ylabel("optimal number of clusters")
plt.title("Number of Clusters by Year")
plt.show()







