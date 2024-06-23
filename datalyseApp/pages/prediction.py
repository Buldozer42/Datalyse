import streamlit as st
import pandas as pd
from front.visualizer import Visualizer
from front.sidebar import menu

st.title("Prédiction")
if 'shared_df' in st.session_state:
    vs = Visualizer(st.session_state['shared_df'])
    
    st.write("### K plus proches voisins (KNN)")
    st.write("Choisissez le nombre de voisins et la colonne cible")
    k = st.number_input("Nombre de voisins", min_value=1, max_value=st.session_state['shared_df'].shape[0], value=5)
    targetKNN = st.selectbox(
        "Choisissez la colonne cible", 
        vs.df.columns, 
        index=None,
        placeholder="Choisissez une colonne...",
        key="targetKNN"
    )

    if targetKNN:
        if st.button("Afficher le KNN", type="primary"):
            vs.showKNN(k, targetKNN)

    st.write("### Régression logistique")
    iterations = st.number_input("Nombre d'itération", min_value=1, max_value=100, value=5)
    targetLR = st.selectbox(
        "Choisissez la colonne cible", 
        vs.df.columns, 
        index=None,
        placeholder="Choisissez une colonne...",
        key="targetLR"
    )
    if targetLR:
        if st.button("Afficher la régression logistique", type="primary"):
            vs.showLogisticRegression(iterations, targetLR)
else:
    st.error("Aucun fichier n'a été importé")

menu()