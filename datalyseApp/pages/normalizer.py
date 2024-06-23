import streamlit as st
import pandas as pd
from front.sidebar import menu
from back.classes.DataNormalizer import DataNormalizer

st.title("Normalisation des données")
if 'shared_df' in st.session_state:
    dn = DataNormalizer()
    st.write('Cette page permet de normaliser les données importées.')
    option = st.selectbox(
        "Veuillez choisir une méthode de normalisation.",
        dn.getMethods(),
        index=None,
        placeholder="Choisissez une méthode...",
    )

    if st.button("Normaliser les données", type="primary"):
        if option:
            if option == 'Min/Max':
                st.session_state['normalize_df'] = dn.MinMax(st.session_state['shared_df'])
            elif option == 'Log Scaling':
                st.session_state['normalize_df'] = dn.logScaling(st.session_state['shared_df'])
            elif option == 'Max Abs Scaler':
                st.session_state['normalize_df'] = dn.maxAbsScaling(st.session_state['shared_df'])
            elif option == 'Robust Scaler':
                st.session_state['normalize_df'] = dn.robustScaling(st.session_state['shared_df'])
            elif option == 'Standard Scaler':
                st.session_state['normalize_df'] = dn.normaliserValuesStandardScaler(st.session_state['shared_df'])
            elif option == 'Z-Score':
                st.session_state['normalize_df'] = dn.normaliserValuesZScore(st.session_state['shared_df'])
            # elif option == 'Quantile Transformer':
            #     st.session_state['normalize_dff'] = dn.quantileTransformer(st.session_state['shared_df'])

            st.session_state['status'] = "df_normalized"
            st.write('Voici les données normalisées:')
            st.write(st.session_state['normalize_df'])

            st.write("Vous pouvez maintenant passer à l'analyse par composantes principales (PCA)")
else:
    st.error("Aucun fichier n'a été importé")

menu()