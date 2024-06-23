from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

# This class allows to apply PCA in the data in a dataset normalized, depending on the method used
class DataPCAManager:

    # First mornaliser values management method
    # Method for dimensionality reduction, noise elimination and data visualization.
    def PCA(self, normalizedData, n_components):
        pca = PCA(n_components=n_components)
        pcaData = pca.fit_transform(normalizedData)
        return pcaData
    
    def visualisationPCA(self, pcaData):
        # Plot PCA
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(x=pcaData[:, 0], y=pcaData[:, 1])
        ax.set_title('PCA Visualization')
        plt.show()
        
    # pcaData doit avoir obligatoirement n_components = 3
    def VisualisationPCA3D(self, pcaData, labels, centers=None, title='Clustering with PCA (3D)'):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(pcaData[:, 0], pcaData[:, 1], pcaData[:, 2], c=labels, cmap='viridis', s=50)
        ax.set_title(title)
        ax.set_xlabel('PCA Feature 1')
        ax.set_ylabel('PCA Feature 2')
        ax.set_zlabel('PCA Feature 3')
        legend = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend)
        
        if centers is not None:
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=100, c='red', marker='X', edgecolor='black', label='Centroids')
            ax.legend()

        plt.show()
        
        
    def Coudes(data, choice) :
        # Calcul des indices CH et DB pour différents nombres de clusters
        max_k = 10
        CH_scores = []
        DB_scores = []
        inertia = []

        for k in range(2, max_k+1):
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(data)
            labels = kmeans.labels_
            
            # Calcul de l'inertie
            inertia.append(kmeans.inertia_)
            
            if choice == 1 :
                # Calcul de l'indice de Calinski-Harabasz
                ch_score = calinski_harabasz_score(data, labels)
                CH_scores.append(ch_score)
                # Plotting CH and DB scores
                plt.figure(figsize=(12, 6))

                plt.subplot(1, 2, 1)
                plt.plot(range(2, max_k+1), CH_scores, marker='o')
                plt.xlabel('Nombre de clusters')
                plt.ylabel('Score de Calinski-Harabasz')
                plt.title('Méthode du coude - Calinski-Harabasz')
            else :
                # Calcul de l'indice de Davies-Bouldin
                db_score = davies_bouldin_score(data, labels)
                DB_scores.append(db_score)

                plt.subplot(1, 2, 2)
                plt.plot(range(2, max_k+1), DB_scores, marker='o')
                plt.xlabel('Nombre de clusters')
                plt.ylabel('Score de Davies-Bouldin')
                plt.title('Méthode du coude - Davies-Bouldin')

                plt.tight_layout()
                plt.show()
