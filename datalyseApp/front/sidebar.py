import streamlit as st

def menu():
    # Show a navigation menu for authenticated users
    st.sidebar.page_link("app.py", label="Accueil")
    if 'status' in st.session_state:
        st.sidebar.page_link("pages/visualisation.py", label="Visualisation")
        st.sidebar.page_link("pages/prediction.py", label="Prédiction")
        st.sidebar.page_link("pages/normalizer.py", label="Normalisation")
        if st.session_state['status'] == "df_normalized":
            st.sidebar.page_link("pages/pca.py", label="PCA")
            if 'pca_df' in st.session_state:
                st.sidebar.page_link("pages/cluster.py", label="Clustering")