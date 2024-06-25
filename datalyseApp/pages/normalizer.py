import streamlit as st
import pandas as pd
from front.sidebar import menu
from back.classes.DataNormalizer import DataNormalizer
from back.classes.MissingValueManager import MissingValueManager

st.title("Normalisation des données")
if 'shared_df' in st.session_state:
    dn = DataNormalizer()
    mvm = MissingValueManager()
    st.write('Cette page permet de normaliser les données importées.')

    if st.button("Remplacer les valeurs catégorielles par des valeurs numériques", type="primary"):
        st.session_state['shared_df'], dict = mvm.replaceString(st.session_state['shared_df'])
        st.write('Les valeurs catégorielles ont été remplacées par des valeurs numériques.')
        st.write(dict)

    columns_options = st.multiselect(
        "Si vous souhaitez supprimer des colonnes, veuillez les sélectionner ici :",
        st.session_state['shared_df'].columns,
        default=None,
        placeholder="Sélectionnez les colonnes à supprimer...",
    )
    if st.button("Supprimer les colonnes", type="primary"):
        if columns_options:
            st.session_state['shared_df'] = dn.removeGivenColumns(st.session_state['shared_df'], columns_options)
            st.write('Les colonnes sélectionnées ont été supprimées.')

    missing_opotion = st.selectbox(
        "Comment souhaitez-vous gérer les valeurs manquantes ?",
        ['Supprimer les lignes', 'Remplacer par la moyenne', 'Remplacer par le mode', 'Remplacer par la médiane', 'Remplacer par KNN'],
        index=None,
        placeholder="Choisissez une méthode...",
    )
    
    if missing_opotion:
        if st.button("Gérer les valeurs manquantes", type="primary"):
            if missing_opotion == 'Supprimer les lignes':
                st.session_state['shared_df'] = mvm.deleteMissingValues(st.session_state['shared_df'])
            elif missing_opotion == 'Remplacer par la moyenne':
                st.session_state['shared_df'] = mvm.fillMissingValues(st.session_state['shared_df'], 'mean')
            elif missing_opotion == 'Remplacer par le mode':
                st.session_state['shared_df'] = mvm.fillMissingValues(st.session_state['shared_df'], 'mode')
            elif missing_opotion == 'Remplacer par la médiane':
                st.session_state['shared_df'] = mvm.fillMissingValues(st.session_state['shared_df'], 'median')
            elif missing_opotion == 'Remplacer par KNN':
                st.session_state['shared_df'] = mvm.fillMissingValuesKNN(st.session_state['shared_df'])

            st.write('Les valeurs manquantes ont été gérées.')

    option = st.selectbox(
        "Veuillez choisir une méthode de normalisation.",
        dn.getMethods(),
        index=None,
        placeholder="Choisissez une méthode...",
    )

    if option:
        if st.button("Normaliser les données", type="primary"):
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

            st.session_state['status'] = "df_normalized"
            st.write('Voici les données normalisées:')
            st.write(st.session_state['normalize_df'])

            st.write("Vous pouvez maintenant passer à l'analyse par composantes principales (PCA)")
else:
    st.error("Aucun fichier n'a été importé")

menu()