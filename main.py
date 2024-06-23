import back.classes.MissingValueManager as mvm
from datalyseApp.back.classes.DataNormalizer import DataNormalizer
import back.classes.ClusteringManager as cm
import back.classes.DataPCAManager as dpm
import pandas as pd

def main():

    # Test the missing value manager
    print('========== MISSING VALUE MANAGER TEST ==========')
    
    mvm1 = mvm.MissingValueManager()
    data = pd.DataFrame(pd.read_csv('./back/mansize.csv', sep=';', header=0))
    
    # Test the deleteMissingValues method
    print('============== DELETE ROWS TEST ==============')
    print('Original data: ')
    print(data)
    
    cleanData = mvm1.deleteMissingValues(data)

    print('Cleaned data: ')
    print(cleanData)



    # Test the fillMissingValues method
    print('============== FILL MISSING VALUES TEST ==============')

    print('Original data: ')
    print(data)

    # Filling data with mean
    filledData = mvm1.fillMissingValues(data, 'mean')
    print('Filled data (mean): ')
    print(filledData)

    # Filling data with mode
    filledData = mvm1.fillMissingValues(data, 'mode')
    print('Filled data (mode): ')
    print(filledData)

    # Filling data with median
    filledData = mvm1.fillMissingValues(data, 'median')
    print('Filled data (median): ')
    print(filledData)

    # Filling data with invalid method
    filledData = mvm1.fillMissingValues(data, 'MyIncredibleMethodWithAwesomeResults')
    print('Filled data (invalid): ')
    print(filledData)

    # Test the fillMissingValuesKNN method
    print('============== FILL MISSING VALUES WITH KNN TEST ==============')
    filledData = mvm1.fillMissingValuesKNN(data)
    print('Filled data (KNN): ')
    print(filledData)

    # Test the Data normaliser
    print('========== DATA NORMALISER TEST ==========')
    
    dn1 = DataNormalizer()

    # First, we need to fill the missing values
    normalisationData = mvm1.fillMissingValues(data, 'mean')

    # Test the Min-Max normalisation method
    print('============== MIN-MAX TEST ==============')

    normalisedData = dn1.MinMax(normalisationData)
    print('Normalised data: ')
    print(normalisedData)


    # Test the normaliserValuesZScore method
    print('============== NORMALISE VALUES WITH Z-SCORE ==============')
    
    filledData = mvm1.fillMissingValues(data, 'mean')
    normalizedData = dn1.normaliserValuesZScore(filledData)
    print('Normalise data with z-score: ')
    print(normalizedData)
    
    
    # Test the PCA manager
    print('============= PCA TEST ============')

    pca1 = dpm.DataPCAManager()
    pcaData = pca1.PCA(normalizedData, 2)

    print("PCA normalilzed data : ", pcaData)
    
    pca1.visualisationPCA(pcaData)
    
    # Test the clustering manager
    print('========== CLUSTERING MANAGER TEST ==========')
    
    cm1 = cm.ClusteringManager()
    data = pd.DataFrame(pd.read_csv('./back/mansize.csv', sep=';', header=0))
    filledData = mvm1.fillMissingValues(data, 'mean')
    normalizedData = dn1.normaliserValuesStandardScaler(filledData)
    
    
    # Test the clusteringKMeans method
    print('============== CLUSTERING VALUES WITH K-MEANS ==============')

    n_clusters = 2 
    random_state = 0 # peut aller haut (test avec 1 000)
    kmeans_labels, kmeans_centers = cm1.clusteringKMeans(normalizedData, n_clusters, random_state)
    print('Clustering data with k-means: ')
    print(kmeans_labels)

    cm1.KMeansVisualisationPCA(pcaData, kmeans_labels, kmeans_centers)
    
    pcaData3 = pca1.PCA(normalizedData, 3)
    pca1.VisualisationPCA3D(pcaData3, kmeans_labels, centers=kmeans_centers, title='K-Means Clustering with PCA (3D)')
    # Compute and display K-Means cluster statistics
    kmeans_stats = cm1.clusterStatistics(normalizedData, kmeans_labels, kmeans_centers)
    cm1.printClusterStatistics(kmeans_stats)


    # Test the clusteringDBSCAN method
    print('============== CLUSTERING VALUES WITH AGGLOMERATION ==============')
    
    
    n_clusters = 3
    agglo_centers, agglo_labels = cm1.clusteringAgglomerative(n_clusters, normalizedData)
    print('Clustering data with agglomerative : ')
    print(agglo_labels)
    
    cm1.AggloVisualisationPCA(pcaData, agglo_labels)
    
    pcaData3 = pca1.PCA(normalizedData, 3)
    pca1.VisualisationPCA3D(pcaData3, agglo_labels, centers=agglo_centers, title='Agglomerative Clustering with PCA (3D)')

    # Compute and display Agglomerative Clustering statistics
    agglo_stats = cm1.clusterStatistics(normalizedData, agglo_labels)
    cm1.printClusterStatistics(agglo_stats)
    
    return
main()