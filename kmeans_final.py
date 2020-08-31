# K-Means Clustering
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sklearn.cluster as cluster
import sklearn.metrics as metrics

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
dataset_num = dataset._get_numeric_data()
df_num = dataset_num.iloc[:,1:4]
#---------------------------------------------------------------------------------------------#
# Feature Scaling
sc_X = StandardScaler()
X = sc_X.fit_transform(df_num)
#---------------------------------------------------------------------------------------------#
# Using the elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# # Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
print('All cluster labels:'+ str(y_kmeans))
kmeans = kmeans.fit(X)
print('Different cluster labels:'+ str(kmeans.labels_))
print('Cluster centroids:'+ str(kmeans.cluster_centers_))


# Creating a new columns & renaming that as cluster
y_kmeans_df = pd.DataFrame(y_kmeans)
y_kmeans_df = y_kmeans_df.rename(columns={0: 'Cluster'})

# Concatenate that column to original dataframe
kmeans_final = pd.concat([dataset, y_kmeans_df], axis = 1)
print('Final dataset with clusters:'+ str(kmeans_final))

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Different cluster labels
labels = KMeans(n_clusters = 5, random_state = 42).fit(X).labels_

# Evaluating silhouette score
sil_score = metrics.silhouette_score(X, labels, metric = "euclidean", sample_size = 10000, random_state = 42)
print(sil_score)
for i in range(4,10):
    labels = KMeans(n_clusters = i, random_state = 42).fit(X).labels_
    print("Silhoutte score for k = " + str(i) + " is " + str(metrics.silhouette_score(X,labels,metric="euclidean",sample_size=1000,random_state=42)))


