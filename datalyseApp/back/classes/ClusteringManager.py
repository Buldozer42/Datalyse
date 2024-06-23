from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# This class allows to cluster the data in a dataset in different ways, depending on the methods used
class ClusteringManager :

    # This method will cluster values using the k-means
    def clusteringKMeans(self, normalizedData, n_clusters, random_state):

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(normalizedData)
        silhouette_avg = silhouette_score(normalizedData, kmeans.labels_)
        return kmeans.labels_, kmeans.cluster_centers_, silhouette_avg
    
    # This method will cluster values using Agglomerative method
    def clusteringAgglomerative(self, n_clusters, normalizedData, linkage='ward'):
        agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit(normalizedData)
        labels = agglo.labels_
        
        unique_labels = np.unique(labels)
        agglo_centers = np.array([normalizedData[labels == label].mean(axis=0) for label in unique_labels])
        silhouette_avg = silhouette_score(normalizedData, labels)
        return agglo_centers, labels, silhouette_avg
       

    # This method will plot the silhouette score of the clusters
    def clusterStatistics(self, data, labels, cluster_centers=None):
        cluster_stats = {}
        unique_labels = np.unique(labels)

        for label in unique_labels:
            cluster_data = data[labels == label]
            cluster_size = len(cluster_data)
            cluster_center = cluster_centers[label] if cluster_centers is not None else np.mean(cluster_data, axis=0)
            cluster_density = cluster_size / np.linalg.norm(cluster_data - cluster_center, axis=1).sum()

            cluster_stats[label] = {
                'size': cluster_size,
                'center': cluster_center,
                'density': cluster_density
            }

        return cluster_stats

    def getMethods(self):
        return ['K-Means', 'Agglomerative']