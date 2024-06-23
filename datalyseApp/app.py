import streamlit as st
from front.fileManager import FileManager
from front.sidebar import menu

st.logo("./datalyseApp/front/images/logo.png", icon_image="./datalyseApp/front/images/logo.png")

st.image("./datalyseApp/front/images/Datalyze.png", use_column_width=True)
st.title("Accueil")
st.write("Bienvenue sur Datalyse, l'application de visualisation de données")

file_manager = FileManager()
file_manager.showFileSelector()

if file_manager.df is not None:
    st.session_state['status'] = "df_loaded"
    st.session_state['shared_df'] = file_manager.df

    if st.button("Visualisation", type="primary"):
        st.switch_page("pages/visualisation.py")
    if st.button("Prédiction", type="primary"):
        st.switch_page("pages/prediction.py")
    if st.button("Normalisation", type="primary"):
        st.switch_page("pages/normalizer.py")

    st.write("### Informations sur le fichier")
    file_manager.showFileInfo()

menu()
