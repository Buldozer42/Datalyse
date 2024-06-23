import streamlit as st
import pandas as pd
from front.visualizer import Visualizer
from front.sidebar import menu

st.title("Visualisation des données")
if 'shared_df' in st.session_state:
    vs = Visualizer(st.session_state['shared_df'])
    vs.showDataFrame()

    st.write("## Histogrammes")
    if st.button("Afficher les histogrammes de toutes les colonnes", type="primary"):
        vs.showHistogram()

    hist_option = st.selectbox(
        "Choisissez une colonne pour afficher son histogramme",
        vs.df.columns,
        index=None,
        placeholder="Choisissez une colonne...",
    )

    if hist_option:
        if st.button("Afficher l'histogramme de la colonne sélectionnée", type="primary"):
            vs.showHistogram([hist_option])

    st.write("## Boîtes à moustaches")
    
    if st.button("Afficher les boîtes à moustaches de toutes les colonnes", type="primary"):
        vs.showBoxplot()

    boxplot_option = st.selectbox(
        "Choisissez une colonne pour afficher sa boîte à moustaches",
        vs.df.columns,
        index=None,
        placeholder="Choisissez une colonne...",
    )

    if boxplot_option:
        if st.button("Afficher le boxplot de la colonne sélectionnée", type="primary"):
            vs.showBoxplot([boxplot_option])
    
    st.write("## Matrice de corrélation")
    if st.button("Afficher la matrice de corrélation", type="primary"):
        vs.showCorrelationMatrix()

else:
    st.error("Aucun fichier n'a été importé")

menu()
