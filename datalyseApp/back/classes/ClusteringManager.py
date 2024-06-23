from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# This class allows to cluster the data in a dataset in different ways, depending on the methods used
class ClusteringManager :

    # First cluster management method
    # This method will cluster values using the k-means
    def clusteringKMeans(self, normalizedData, n_clusters, random_state):

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(normalizedData)
        silhouette_avg = silhouette_score(normalizedData, kmeans.labels_)
        return kmeans.labels_, kmeans.cluster_centers_, silhouette_avg
     
    # First cluster visualisation method
    # This method will see the clustering values using the k-means with PCA
    # pcaData doit avoir obligatoirement n_components = 2
    def kMeansVisualisationPCA(self, pcaData, kmeans_labels, kmeans_centers):
        # Visualize K-Means clustering with PCA
        centroids = kmeans_centers

        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=pcaData[:, 0], y=pcaData[:, 1], hue=kmeans_labels, palette='viridis', s=50)
        
        plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='red', marker='X', label='Centroids')
        
        plt.title('K-Means Clustering with PCA')
        plt.xlabel('PCA Feature 1')
        plt.ylabel('PCA Feature 2')
        plt.legend(title='Cluster')
        plt.show()


    # Second cluster management method
    # This method will cluster values using Agglomerative method
    def clusteringAgglomerative(self, n_clusters, normalizedData, linkage='ward'):
        agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit(normalizedData)
        labels = agglo.labels_
        
        unique_labels = np.unique(labels)
        agglo_centers = np.array([normalizedData[labels == label].mean(axis=0) for label in unique_labels])
        silhouette_avg = silhouette_score(normalizedData, labels)
        return agglo_centers, labels, silhouette_avg


    # First cluster visualisation method
    # This method will see the clustering values using the k-means with PCA
    # pcaData doit avoir obligatoirement n_components = 2
    def aggloVisualisationPCA(self, pcaData, agglo_labels):
        # Calculer les centroids
        unique_labels = np.unique(agglo_labels)
        centroids = np.array([pcaData[agglo_labels == label].mean(axis=0) for label in unique_labels])

        # Visualiser le clustering agglomératif avec PCA
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=pcaData[:, 0], y=pcaData[:, 1], hue=agglo_labels, palette='viridis', s=50)

        # Ajouter les centroids au plot
        plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='red', marker='X', label='Centroids')
        
        plt.title('Agglomerative Clustering with PCA')
        plt.xlabel('PCA Feature 1')
        plt.ylabel('PCA Feature 2')
        plt.legend(title='Cluster')
        plt.show()
        

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

    def printClusterStatistics(self, cluster_stats):
        for label, stats in cluster_stats.items():
            print(f"Cluster {label}:")
            print(f"  Size: {stats['size']}")
            print(f"  Center: {stats['center']}")
            print(f"  Density: {stats['density']:.4f}")

    def getMethods(self):
        return ['K-Means', 'Agglomerative']