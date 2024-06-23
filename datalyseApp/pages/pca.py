import streamlit as st
import pandas as pd
from front.sidebar import menu
from back.classes.DataPCAManager import DataPCAManager
from front.visualizer import Visualizer

st.title("Analyse par composantes principales (PCA)")
if 'shared_df' in st.session_state :
    if st.session_state['status'] == "df_normalized":
        pca_manager = DataPCAManager()
        vs = Visualizer(st.session_state['shared_df'])

        st.write('Cette page vous permet de réaliser une réduction de dimensions avec PCA avant de pouvoir réaliser un clustering')   
        st.write("Choisissez le nombre de composantes principales, c'est-à-dire le nombre de dimensions sur lesquelles vous souhaitez réduire vos données")    
       
        n_components = st.number_input('Nombre de composantes principales :', min_value=1, max_value=st.session_state['shared_df'].shape[1], value=3)
        if st.button("Réduire les dimensions", type="primary"):
            if n_components:
                st.session_state['pca_df'] = pca_manager.PCA(st.session_state['shared_df'], n_components)
                st.session_state['n_components'] = n_components
                st.write("Vous pouvez passer aux clustering (via le menu) ou visualiser les données réduites avec PCA")
                st.write('Voici les données réduites avec PCA :')
                st.write(st.session_state['pca_df'])
                st.write('Visualisation des données réduites avec PCA :')
                vs.__setattr__('pcaDf', st.session_state['pca_df'])
                vs.showPCA()
                if n_components >= 3:
                    vs.showPCA3D()
    else:
        st.error("Les données n'ont pas été normalisées")
else:
    st.error("Aucun fichier n'a été importé")

menu()