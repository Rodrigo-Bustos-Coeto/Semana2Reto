# Note: it is 'sklearn', not 'scikit-learn'
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd

columnas = ['Carga', 'Masa', 'TiempoVuelo', 'VelMax', 'PosFinalX', 'Clase']
df = pd.read_csv('datosCelulas.csv', names=columnas, header=0)

X = df[['Carga', 'TiempoVuelo', 'VelMax', 'PosFinalX']]
y = df['Clase']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=2,
    init='k-means++',
    n_init=7,
    max_iter=100,
    tol=1e-1,
    random_state=18
)

y_pred_km = kmeans.fit_predict(X_scaled)

comparison = pd.crosstab(y, y_pred_km, rownames=['Actual'], colnames=['Cluster'])
print("--- Comparison Table ---")
print(comparison)

# Check the coordinates of the cluster centers
print("\n--- Cluster Centers (Scaled) ---")
print(kmeans.cluster_centers_)


counts = pd.Series(y_pred_km).value_counts().sort_index()

# 2. Create the plot
plt.figure(figsize=(8, 5))
bars = plt.bar(['Cluster 0', 'Cluster 1'], counts.values, color=['skyblue', 'lightgreen'])

# 3. Add titles and labels
plt.title('Frecuencia de Clusters - K-Means')
plt.xlabel('Clusters')
plt.ylabel('Número de Células')

# Add the exact count on top of each bar
plt.bar_label(bars, padding=3)

# Display the plot
plt.show()

