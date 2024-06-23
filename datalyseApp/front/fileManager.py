import streamlit as st
import pandas as pd

# This class allows to manage the file uploaded by the user
class FileManager:
    def __init__(self):
        self.uploaded_file = None
        self.df = None
        self.separator = None
    
    def showFileSelector(self):
        self.separator = st.text_input("Donnez le séparateur de votre fichier CSV", ";")
        st.write("Séparateur choisi :", self.separator)

        # Upload file
        self.uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=["csv"])

        # Check if the file is uploaded
        if self.uploaded_file is not None and self.separator:
            try:                
                # To read file as pandas dataframe
                self.df = pd.read_csv(self.uploaded_file, sep=self.separator)
                
            except Exception as e:
                st.error("Le fichier uploadé n'est pas un fichier CSV valide ou vous n'avez pas donné le séparateur de votre fichier CSV")
                st.error(f"Erreur: {e}")

    def showFileInfo(self):
        st.write("**5 premières lignes du dataframe :**",self.df.head())

        st.write("**5 dernières lignes du dataframe :**",self.df.tail())

        st.write("### Informations sur le dataframe")
                
        st.write("**Noms des colonnes :**")
        st.write(self.df.columns.tolist())
                
        st.write("**Nombre de lignes et de colonnes :**")
        st.write(f"Lignes : {self.df.shape[0]}")
        st.write(f"Colonnes : {self.df.shape[1]}")
                
        st.write("**Nombre de valeurs manquantes par colonne :**")
        st.write(self.df.isnull().sum())

        st.write("**Statistiques descriptives :**", self.df.describe())