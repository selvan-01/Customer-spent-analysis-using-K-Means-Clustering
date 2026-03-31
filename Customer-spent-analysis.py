# ============================================
# 📊 Customer Spend Analysis using K-Means Clustering
# ============================================

# --------- 1. Import Required Libraries ---------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# --------- 2. Load Dataset ---------
# Upload dataset from local system (Google Colab)
from google.colab import files
uploaded = files.upload()

# Read dataset
dataset = pd.read_csv('dataset.csv')


# --------- 3. Dataset Overview ---------
print("🔹 Dataset Shape:", dataset.shape)
print("\n🔹 Dataset Description:\n", dataset.describe())
print("\n🔹 First 5 Rows:\n", dataset.head())


# --------- 4. Feature Selection ---------
# Extract Income and Spend columns
Income = dataset['INCOME'].values
Spend = dataset['SPEND'].values

# Combine both features into a single array
X = np.array(list(zip(Income, Spend)))


# --------- 5. Finding Optimal Number of Clusters (Elbow Method) ---------
from sklearn.cluster import KMeans

wcss = []  # Within Cluster Sum of Squares

# Run K-Means for different cluster values (1 to 10)
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  # Store inertia (WCSS)

# Plot Elbow Graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('📌 Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()


# --------- 6. Apply K-Means Clustering (K = 4) ---------
kmeans_model = KMeans(n_clusters=4, random_state=0)

# Predict cluster labels
y_means = kmeans_model.fit_predict(X)


# --------- 7. Visualizing Clusters ---------
plt.figure(figsize=(8, 6))

# Plot each cluster with different colors
plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1],
            s=50, c='brown', label='Cluster 1')

plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1],
            s=50, c='blue', label='Cluster 2')

plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1],
            s=50, c='green', label='Cluster 3')

plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1],
            s=50, c='cyan', label='Cluster 4')


# Plot centroids
plt.scatter(kmeans_model.cluster_centers_[:, 0],
            kmeans_model.cluster_centers_[:, 1],
            s=100, c='red', marker='s', label='Centroids')


# Labels & Title
plt.title('💰 Customer Income vs Spending Analysis')
plt.xlabel('Income')
plt.ylabel('Spending')
plt.legend()
plt.grid(True)

plt.show()


# --------- 8. Cluster Insights ---------
"""
Cluster Interpretation:

Cluster 1 → Medium Income, Low Spending
Cluster 2 → High Income, High Spending
Cluster 3 → Low Income Customers
Cluster 4 → Medium Income, High Spending

💡 Business Insight:
- Target Cluster 2 for premium products
- Improve engagement for Cluster 1
- Offer discounts to Cluster 3
- Upsell to Cluster 4
"""