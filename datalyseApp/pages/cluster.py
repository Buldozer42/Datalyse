import streamlit as st
import pandas as pd
from front.sidebar import menu
from back.classes.ClusteringManager import ClusteringManager
from front.visualizer import Visualizer

st.title("Clustering")
if 'shared_df' in st.session_state :
    if 'pca_df' in st.session_state:
        cm = ClusteringManager()
        vs = Visualizer(pcaDf=st.session_state['pca_df'])
        st.write('Cette page permet de faire du clustering sur les données importées.')
        option = st.selectbox(
            "Veuillez choisir une méthode de clustering.",
            cm.getMethods(),
            index=None,
            placeholder="Choisissez une méthode...",
        )
        if option == 'K-Means':
            st.write('Vous avez choisi la méthode K-Means')
            n_clusters = st.number_input('Nombre de clusters', min_value=2, max_value=(st.session_state['pca_df'].shape[0]-1), value=2)
            random_state = st.number_input('Random state', min_value=0, value=0)
            if st.button("Lancer le clustering", type="primary"):
                kmeans_labels, kmeans_centers, silhouette_knn = cm.clusteringKMeans(st.session_state['pca_df'], n_clusters, random_state)
                kmeans_stats = cm.clusterStatistics(st.session_state['pca_df'], kmeans_labels, kmeans_centers)
                st.write('Silhouette :' , silhouette_knn)
                for label, stats in kmeans_stats.items():
                    st.write(f"**Cluster {label}**")
                    st.write(f"- Size: {stats['size']}")
                    st.write(f"- Center: {stats['center']}")
                    st.write(f"- Density: {stats['density']:.4f}")

                st.write('Visualisation du clustering avec PCA :')
                vs.showKmeansClustering(kmeans_labels, kmeans_centers)
                if st.session_state['n_components'] >= 3:
                    vs.showPCA3D(kmeans_labels, centers=kmeans_centers)
                
        if option == 'Agglomerative':
            st.write('Vous avez choisi la méthode Agglomerative')
            n_clusters = st.number_input('Nombre de clusters', min_value=2, max_value=(st.session_state['pca_df'].shape[0]-1), value=2)
            if st.button("Lancer le clustering", type="primary"):
                agglo_centers, agglo_labels, silhouette_agglo = cm.clusteringAgglomerative(n_clusters, st.session_state['pca_df'])
                agglo_stats = cm.clusterStatistics(st.session_state['pca_df'], agglo_labels, agglo_centers)
                st.write('Silhouette :' , silhouette_agglo)
                for label, stats in agglo_stats.items():
                    st.write(f"**Cluster {label}**")
                    st.write(f"- Size: {stats['size']}")
                    st.write(f"- Center: {stats['center']}")
                    st.write(f"- Density: {stats['density']:.4f}")
                
                st.write('Visualisation du clustering avec PCA :')
                vs.showAgglomerativeClustering(agglo_labels)
                if st.session_state['n_components'] >= 3:
                    vs.showPCA3D(agglo_labels, centers=agglo_centers)
    else:
        st.error("Les données n'ont pas été réduites avec PCA")
else:
    st.error("Aucun fichier n'a été importé")

menu()