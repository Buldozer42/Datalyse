from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# This class allows to use some prediction algorithms (classification and regression)
class Predictor:

    # KNN algorithm (here, k is the number of neighbors, and target is the target column name)
    def knnPrediction(self, data, k, target):
        data = self.floatToIntConversion(data)
        X = data.drop(target, axis=1)
        y = data[target]

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X, y)

        # Number of features and samples used to fit the model, as well as model score
        nfeatures = knn.n_features_in_
        nsamples = knn.n_samples_fit_
        score = knn.score(X, y)
        predict = knn.predict(X)

        return knn, nfeatures, nsamples, score, predict
    

    # Logistic regression (here, target is the target column name, iterations is the number of iterations to perform)
    def logisticRegressionPrediction(self, data, target, iterations):
        data = self.floatToIntConversion(data)
        X = data.drop(target, axis=1)
        y = data[target]

        # Create the model
        logreg = LogisticRegression(max_iter=iterations)
        logreg.fit(X, y)

        # Number of features and samples used to fit the model, as well as model score
        nfeatures = logreg.n_features_in_
        score = logreg.score(X, y)
        predict = logreg.predict(X)

        return logreg, nfeatures, score, predict

    # Allows to convert every float value in df to an int value
    def floatToIntConversion(self,data):
        return data.astype(int)