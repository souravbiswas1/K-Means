# Importing necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
import sklearn.cluster as cluster
from scipy.spatial.distance import cdist
import sklearn.metrics as metrics
# import cluster_profiles as cluster_profiles

# Setup a working directory
data_dir = 'D:\\Zencode\\Projects\\POC\\10.K-Means'
os.chdir(data_dir)
data = pd.read_csv("Mall_Customers.csv")

# Identify the numeric variables
data_num = data._get_numeric_data()
data_num = data_num.iloc[:,1:4]

# Scale the data, using pandas
def scale(x):
	return (x - np.mean(x)) / np.std(x)
data_scaled = data_num.apply(scale, axis = 0)

# Scale the data using sklearn
dat_scaled = preprocessing.scale(data_num, axis = 0)
print ("Type of output is "+ str(type(dat_scaled)))
print ("Shape of the object is "+ str(dat_scaled.shape))

# Create a cluster model
kmeans = cluster.KMeans(n_clusters = 3, init = "k-means++", random_state = 42)
kmeans = kmeans.fit(dat_scaled)
print(kmeans.labels_)
print(kmeans.cluster_centers_)

# Elbow method
K = range(1,20)
wss = []
for k in K:
    kmeans = cluster.KMeans(n_clusters = k,init = "k-means++", random_state = 42)
    kmeans.fit(dat_scaled)
    wss.append(sum(np.min(cdist(dat_scaled, kmeans.cluster_centers_, 'euclidean'), axis = 1)) / dat_scaled.shape[0])

# Plotting the elbow method
plt.plot(K, wss, 'bx')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Selecting k with the Elbow Method')
plt.show()

# Different cluster labels
labels = cluster.KMeans(n_clusters = 6, random_state = 42).fit(dat_scaled).labels_

# Evaluating silhouette score
sil_score = metrics.silhouette_score(dat_scaled, labels, metric = "euclidean", sample_size = 10000, random_state = 42)
print(sil_score)
for i in range(4,10):
    labels = cluster.KMeans(n_clusters = i, random_state = 42).fit(dat_scaled).labels_
    print("Silhoutte score for k = " + str(i) + " is " + str(metrics.silhouette_score(dat_scaled,labels,metric="euclidean",sample_size=1000,random_state=42)))

# Let's look for profiles for 5 & 6 clusters
# kmeans = cluster.KMeans(n_clusters = 6,random_state = 42).fit(dat_scaled)
# cluster_pro = cluster_profiles.get_zprofiles(data = data_num.copy(), kmeans = kmeans)
# print(cluster_pro)
