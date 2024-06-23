import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler

# This class allows to normalise the data in a dataset in different ways, depending on the methods used
class DataNormalizer:

    # This method normalises the data using the Min-Max normalisation method
    def MinMax(self, data):
        scaler = MinMaxScaler()
        return self.returnData(scaler.fit_transform(data), data.columns)

    # This method normalises the data using the Log scaling normalisation method
    def logScaling(self, data):
        transformer = FunctionTransformer(np.log1p)
        return self.returnData(transformer.fit_transform(data), data.columns)
    
    # This method normalises the data using the Max-Abs scaling normalisation method
    # It scales each feature by its maximum absolute value, values end up between -1 and 1
    def maxAbsScaling(self, data):
        transformer = MaxAbsScaler().fit(data)
        return self.returnData(transformer.transform(data), data.columns)
    
    # This method normalises the data using the Robust scaling normalisation method
    def robustScaling(self, data):
        transformer = RobustScaler().fit(data)
        return self.returnData(transformer.transform(data), data.columns)

    # This method will normalise values using the standard scaler
    def normaliserValuesStandardScaler(self, filledData):
        scaler = StandardScaler()
        return self.returnData(scaler.fit_transform(filledData), filledData.columns)

    # This method will normalise values using the z-score
    def normaliserValuesZScore(self, data):
        return self.returnData(stats.zscore(data), data.columns)
    
    # This method will normalise values using the quantile transformer
    def quantileTransformer(self, data):
        transformer = QuantileTransformer(n_quantiles=10, random_state=0)
        return self.returnData(transformer.fit_transform(data), data.columns)
     
    def returnData(self, data, columns):
        return pd.DataFrame(data, columns=columns)    
    
    def getMethods(self):
        return ['Min/Max', 'Log Scaling', 'Max Abs Scaler', 'Robust Scaler', 'Standard Scaler','Z-Score']