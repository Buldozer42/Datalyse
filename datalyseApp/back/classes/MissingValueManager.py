import pandas as pd
from sklearn.impute import KNNImputer

# This class allows to manage missing values in a dataset in different ways
class MissingValueManager:

    # This method will delete all the rows with missing values
    def deleteMissingValues(self, data):
        return data.dropna()
    
    # Here, we will fill the missing values using the mean, mode or median of each column
    ## method: 'mean', 'mode' or 'median', allows to choose the method to fill the missing values
    def fillMissingValues(self, data, method):
        match method:
            case 'mean':
                # Fills missing value with the mean of each column, rounded to 2 decimal
                return data.fillna(round(data.mean(), 2))
            
            case 'mode':
                # Fills missing value with the mode of each column
                return data.fillna(data.mode().iloc[0])
            
            case 'median':
                # Fills missing value with the median of each column
                return data.fillna(data.median())
            
            case _:
                print('Invalid method')
                return data
        return
    
    # This method will fill the missing values using the KNN algorithm
    def fillMissingValuesKNN(self, data):

        # Create the KNN imputer, using 2 neighbors
        imputer = KNNImputer(n_neighbors=2)
        imputer.fit(data)
        knnData = imputer.transform(data)

        # Put the data back into a DataFrame
        knnData = pd.DataFrame(knnData, columns=data.columns)

        return knnData
    
    # This method will replace the string values with numerical values
    def replaceString(self, data):
        mapping_dict = {}
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col], mapping = pd.factorize(data[col])
                mapping_dict[col] = list(mapping)
        return data, mapping_dict