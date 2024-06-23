import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from back.classes.Predictor import Predictor

# This class allows to visualize the data in a dataset. It handle all graphical representations in the app
class Visualizer:
    def __init__(self, dataframe=None, pcaDf=None):
        self.df = dataframe
        self.pcaDf = pcaDf
        st.set_option('deprecation.showPyplotGlobalUse', False)

    def showDataFrame(self):
        st.write('Données : ')
        st.write(self.df) 

    def showHistogram(self, columns=None):
        if columns is None:
            columns = self.df.columns
        
        for col in columns:
            st.write('Histogramme de la colonne : ', col)
            sns.histplot(self.df[col])
            st.pyplot()

    def showCorrelationMatrix(self):
        st.write('Matrice de corrélation : ')
        sns.heatmap(self.df.corr(), annot=True)
        st.pyplot()

    def showBoxplot(self, columns=None):
        if columns is None:
            columns = self.df.columns
        
        for col in columns:
            st.write('Boxplot de la colonne : ', col)
            sns.boxplot(x=self.df[col])
            st.pyplot()

    def showKNN(self, k, target):
        predictor = Predictor()
        knn, nfeatures, nsamples, score, predict = predictor.knnPrediction(self.df,k, target)
        st.write(f"Nombre de features : {nfeatures}")
        st.write(f"Nombre de samples : {nsamples}")
        st.write(f"Score du modèle : {score}")

        st.write('Prédiction : ')
        st.write(predict)
        st.write('Valeurs réelles : ')
        st.write(self.df[target])

        st.write('Visualisation du modèle ( cette étape peut prendre du temps ) :')
        sns.pairplot(self.df, hue=target)
        st.pyplot()

    def showLogisticRegression(self, iterations, target):
        predictor = Predictor()
        logreg, nfeatures, score, predict = predictor.logisticRegressionPrediction(self.df, target, iterations)
        st.write(f"Nombre de features : {nfeatures}")
        st.write(f"Score du modèle : {score}")

        st.write('Prédiction : ')
        st.write(predict)
        st.write('Valeurs réelles : ')
        st.write(self.df[target])
        st.write('Score du modèle : ')
        st.write(score)

        st.write('Visualisation du modèle ( cette étape peut prendre du temps ) :')
        sns.pairplot(self.df, hue=target)
        st.pyplot()

    def showPCA(self):
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(x=self.pcaDf [:, 0], y=self.pcaDf [:, 1])
        ax.set_title('PCA Visualization')
        st.pyplot()

    def showPCA3D(self, labels=None, centers=None): 
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        if labels is None:
            labels = self.pcaDf [:, 2]
        
        scatter = ax.scatter(self.pcaDf [:, 0], self.pcaDf [:, 1], self.pcaDf [:, 2], c=labels, cmap='viridis', s=50)
        ax.set_title('PCA (3D)')
        ax.set_xlabel('Composante 1')
        ax.set_ylabel('Composante 2')
        ax.set_zlabel('Composante 3')
        legend = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend)
        
        if centers is not None:
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=100, c='red', marker='X', edgecolor='black', label='Centroids')
            ax.legend()

        st.pyplot()

    def showKmeansClustering(self, kmeans_labels, kmeans_centers):
        self.__showClustering(kmeans_labels, kmeans_centers)

    def showAgglomerativeClustering(self, agglo_labels):
        unique_labels = pd.unique(agglo_labels)
        centroids = np.array([self.pcaDf[agglo_labels == label].mean(axis=0) for label in unique_labels]) 
        self.__showClustering(agglo_labels, centroids)

    def __showClustering(self, labels, centroids):
        sns.scatterplot(x=self.pcaDf[:, 0], y=self.pcaDf[:, 1], hue=labels, palette='viridis', s=50)
        sns.scatterplot(x=centroids[:, 0], y=centroids[:, 1], s=100, c='red', marker='X', label='Centroids')
        st.pyplot()

    def showElbow(self, scores, max_k):
        sns.lineplot(x=range(2, max_k), y=scores)
        st.pyplot()