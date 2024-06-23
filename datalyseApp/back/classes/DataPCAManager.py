from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

# This class allows to apply PCA in the data in a dataset normalized, depending on the method used
class DataPCAManager:

    # Method for dimensionality reduction, noise elimination and data visualization.
    def PCA(self, normalizedData, n_components):
        pca = PCA(n_components=n_components)
        pcaData = pca.fit_transform(normalizedData)
        return pcaData
    
    # Method that apply the elbow method to find the optimal number of clusters
    # index = 1 for calinski_harabasz_score and index = 2 for davies_bouldin_score
    def elbows(self, normalizedData, index):
        scores = []
        max_k = 10
        for k in range(2, max_k):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(normalizedData)
            if index == 1:
                score = calinski_harabasz_score(normalizedData, kmeans.labels_)
            if index == 2:
                score = davies_bouldin_score(normalizedData, kmeans.labels_)
            scores.append(score)
        return scores, max_k